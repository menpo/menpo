import abc


class FeatureBasedMotionModel(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def thread(self, centers, q, V):
        pass

    @abc.abstractmethod
    def revert(self, centers, u0, V):
        pass

    @abc.abstractmethod
    def calculate_jacobian(self):
        pass


class FeatureBasedTPS(FeatureBasedMotionModel):

    def __init__(self, centers, u0):
        self.centers = centers
        self.u0 = u0

    @property
    def warp_p(self):
        E = self.__E()
        M = np.linalg.inv(E)
        return M.dot(self.ui)

    def compose_warp(self, delta):
        uprime = self.u0 - delta
        u = self.revert(uprime)
        self.ui = self.thread(u)

    def calculate_jacobian(self, q):
        l = self.centers.shape[0]
        E = self.__E()
        yotta_q = self.__yotta_q(q)
        K = yotta_q.T.dot(E)

        zeros = np.zeros([1, l + 3])
        J_left = np.vstack([K, zeros])
        J_right = np.vstack([zeros, K])

        return np.hstack([J_left, J_right])

    def thread(self, v, vprime):
        E = self.__E()
        yotta_q = self.__yotta_q(v)
        return E.T.dot(yotta_q).dot(vprime)[:-3]

    def revert(self, V):
        E = self.__E()
        yotta_q = self.__yotta_q(V)
        M = np.linalg.inv(E.T.dot(yotta_q)[:-3])
        return M.dot(self.u0)

    def __yotta_q(self, q):
        l = self.centers.shape[0]  # Number of center points
        m = q.shape[0]
        yotta_q = np.zeros([l + 3, m])

        for i in xrange(m):
            for j in xrange(l):
                r = np.sum((q[i, :] - self.centers[j, :]) ** 2)
                yotta_q[j, i] = self.__rho(r)
            yotta_q[-3:-1, i] = q[i, :]
            yotta_q[-1:, i] = 1

        return yotta_q

    def __rho(self, r):
        """
        Radial basis function for TPS.
        """
        if r == 0:
            return 1
        else:
            return r ** 2 * (np.log(r ** 2))

    def __E_lambda(self, l_lambda=0.0001):
        l = self.centers.shape[0]  # Number of center points
        Q = np.hstack([self.centers, np.ones([l, 1])])
        N_lambda = l_lambda * np.eye(l)  # N + lambda * eye

        for i in xrange(l):
            for j in xrange(l):
                if i != j:
                    r = np.sum((self.centers[i, :] - self.centers[j, :]) ** 2)
                    N_lambda[i, j] = self.__rho(r)

        # (Q^T N_lambda^-1 Q)^-1 Q^T N_lambda^-1
        E_part = np.linalg.inv(Q.T.dot(np.linalg.inv(N_lambda).dot(Q))).dot(
            Q.T.dot(np.linalg.inv(N_lambda)))

        # [N_lambda^-1 (I - Q E_part); E_part]
        return np.vstack([np.linalg.inv(N_lambda).dot((np.eye(l) - Q.dot(
            E_part))), E_part])

    def __E(self, l_lambda=0.0001):
        l = self.centers.shape[0]  # Number of center points
        Q = np.hstack([self.centers, np.ones([l, 1])])
        N_lambda = l_lambda * np.eye(l)

        for i in xrange(l):
            for j in xrange(l):
                if i != j:
                    r = np.sum((self.centers[i, :] - self.centers[j, :]) ** 2)
                    N_lambda[i, j] = self.__rho(r)

        M = np.hstack([N_lambda, Q])
        M = np.vstack([M, np.hstack([Q.T, np.zeros([3, 3])])])

        return np.linalg.inv(M)