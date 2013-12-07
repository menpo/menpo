
# 1. Prerequisites
To get started you will need two things

- [XQuartz](http://xquartz.macosforge.org/landing/), an open source X11 implimentation for-  OS X
- [Homebrew](http://brew.sh/), the missing package manager for OS X

Ensure that you have both set up before starting this script. In particular check what `brew doctor` has to say. You should find

    >> brew doctor
    Your system is ready to brew.
    
If not, Google the problems away until the doctor says everything is fine! If you proceed against the doctors advice things may not work in unexpected ways. You have been warned..

**Side note** - *already know what you are doing? Skip to the end for scripts..*

# 2. Installing system libraries and tools

We are now going to use `brew` to install all the system libraries and tools we need for PyBug. If any of the following fail to install correctly, please raise an Issue at [github.com/ibug/pybug](http://github.com/ibug/pybug).

Get the latest git, a clean version of Python, and all the dependencies needed for the base Scientific Python toolchain


    >> brew install git python cmake gfortran pkg-config readline libpng libjpeg freetype


Secondly, get some PyBug specific stuff.

    >> brew install boost assimp pyqt

Our final `brew` dependency is `VTK5`. We need to tap some other repos to install this (as it's an *old version* of a particularly *scientific* package - see [here](https://github.com/mxcl/homebrew/wiki/Homebrew-0.9) for a bit more info on what tap does).

    >> brew tap homebrew/versions
    >> brew tap homebrew/science
    >> brew install vtk5

That's it, everything else is now done in Python. Run a final

    >> brew doctor
    
and check everything is fine before proceeding. If you run

    >> brew list
    
you should see all the packages above listed.

# 3. Setting up a scientific Python install on OS X

Homebrew now has it's own Python install in `/usr/local` which is totally seperate to the system Python that ships with OS X. The great thing about this is that we can install Python packages globally without fear of damaging OS X. *Global Python* now actually means *Homebrew's Python*  - so if anything goes badly wrong we can just

    >> brew uninstall python
    >> brew install python

and reinstall our global Python packages. This is a nice advantage of OS X over Linux, where we would have to create an isolated environment for all of this stuff (otherwise we would have to invoke all these commands with `sudo` and clobber the system Python - which is **very bad!**).


Let's get going by checking pip, setuptools and virtualenvwrapper
are up to date.

    >> pip install --upgrade pip setuptools virtualenvwrapper
    
Next, we can grab the IPython notebook and the scientific tools we want (including some docs tools at the end)

    >> pip install --upgrade 'ipython[notebook]' cython numpy scipy matplotlib pillow sphinx numpydoc nose

Running

    >> pip list

should show all the above packages are installed in the global
namespace. If not, raise an issue!

At this point you have a Matlab-esque Python setup ready to go. Check

    >> ipython notebook
   
works as expected.

# 4. Creating a virutalenv for PyBug development

Now we want to get hold of our code and install it for use. Installing PyBug at this point is super simple - we *could* just do:

    >> pip install git+git@github.com:ibug/pybug.git#egg=pybug
    
(This is just like the pip commands above, only with some extra syntax to tell pip to get the code from a remote git repository.)

However we also want the ability to develop - change files, change git branches, commit new code, and issue pull requests. If we just starting messing around with the python files installed in Homebrew's Python folder, we would quickly be in a world of trouble.

Luckily, Python resolved this problem long ago with virtualenv. Virtualenv creates isolated Python installs where we can mess around to our hearts content without fear of breaking anything. If you don't know anything about virutalenv, you should probably have a read up on it and come back here.

Virtualenv is the tool that allows us to do all this clever work, but it's a bit of a pain to use directly. Thankfully, virtualenvwrapper makes working with virtualenv a breeze. Again, have a quick read up on virutalenvwrapper and pop back here. 

You already have everything you need to use virtualenvwrapper installed, but it may not be accessable to you yet. Unfortunatley, how we enable virtualenvwrapper is dependent on the shell you are using. If you are using the default OS X shell (`/bin/bash`) then all you need to do is add

    if [[ -s "/usr/local/bin/virtualenvwrapper.sh" ]]; then
      source "/usr/local/bin/virtualenvwrapper.sh"
    fi
    
to the end of your `~/.bashrc` file. (If you don't have one yet, then make it with just those lines in!) To reload the file, type

    >> . ~/.bashrc
    
and you should find you can use the virutalenvwrapper tools. If you are on a different shell I'll assume you know how to add a similar source statement to the relevent startup dotfile!

To check everything is fine you should be able to make a virtualenv for PyBug development

    >> mkvirutalenv pybug
     
This will create a folder at `~/.virtualenvs/pybug/` which is **where we will do all of our development**.

By default, virtualenv's are totally isolated from the global Python. We want to reuse all the tools we installed globally so we can run

    (pybug) >> toggleglobalsitepackages
    
to allow global Python packages to be visable in the virutalenv. If you run

    (pybug) >> pip list
    
now you will see all the previously installed global tools. Adding the `-l` flag allows us to just see what's in the virtualenv - you should see there isn't much at all

    (pybug) >> pip list -l
    
Notice the `(pybug)` on the left hand side of the prompt. This reminds us that the pybug virutalenv is active. If we issue the deactivate command

    (pybug) >> deactivate
    >>
    
you see that the prompt goes back to normal. To renable pybug, we just use the `workon` command

    >> workon pybug
    (pybug) >>
    
Issuing a pip install command with the pybug virtualenv active will install the python package to `~/.virtualenvs/pybug` as opposed to `/usr/local/`. 

The final piece of the puzzle is that we would like to be able to edit the package we want to install easily. To do so, we just pass the `-e` flag to pip

    >> pip install -e git+git@github.com:ibug/pybug.git#egg=pybug

Packages installed with the editable flag are installed in the `src` folder under the virtualenv, so now we can find pybug at 
    
    ~/.virtualenvs/pybug/src/pybug/
    
This is an actual git repository, and is wired up as a true Python package! What this means is, as long as the `pybug` virutalenv is active, I can open an IPython notebook **anywhere** on my system, import pybug, and **this folder is what is loaded**. If I made a change, say add `print 'hi there!'` to `~/.virtualenvs/pybug/src/pybug/pybug/__init__.py`, restart the IPython notebook, and `import pybug` I should get a warm greeting. 

Now you are all set! Just setup your IDE or whatever (we strongly recommend PyCharm) to point to `~/.virtualenvs/pybug/src/pybug/`, make changes, reload your Python interpreter, and test.


# Install scripts

If you already know what you are doing and just want to automate the process, run the two shell scripts in this folder in order. If all goes well you will have all the brew and pip dependences installed globally. Setup virtualenv however you want and just grab pybug with


    >> pip install -e git+git@github.com:yourrepo/pybug.git#egg=pybug

