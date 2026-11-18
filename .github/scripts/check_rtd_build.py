#!/usr/bin/env python3
"""Wait for Read the Docs to build a released version and fail if the build fails.

Read the Docs reads its configuration (.readthedocs.yaml) from the tagged commit, so a tag with a
broken docs setup cannot be repaired afterwards (the tag must not move). This is run after a release
is published to make that visible straight away instead of leaving `stable` silently outdated.

With --preflight it is run BEFORE publishing instead: Read the Docs builds `latest` from every push to
the default branch and reads the same configuration from that commit, so a successful `latest` build of
the commit about to be released shows that the tag's docs will build (a failed one blocks the release).

Only the public API of the (public) project is used, no token is needed.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta

API = "https://app.readthedocs.org/api/v3/projects/{project}/versions/{version}/"
# Read the Docs only activates/builds `stable` from final releases
PRERELEASE = re.compile(r"(\d(a|b|rc)\d+|\.dev\d+)$")


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "rtd-release-check"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def parse(timestamp):
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default=os.environ.get("GITHUB_REPOSITORY", "menpo/menpo").split("/")[-1],
        help="Read the Docs project slug (default: the repository name)",
    )
    parser.add_argument(
        "--version", default="stable", help="Read the Docs version to check"
    )
    parser.add_argument("--tag", help="release tag; pre-releases are skipped")
    parser.add_argument(
        "--sha", help="commit the version must point at (the tagged commit)"
    )
    parser.add_argument(
        "--since", required=True, help="ISO time: only builds created after it count"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="before publishing: require the newest `latest` build to have succeeded for --sha",
    )
    parser.add_argument(
        "--timeout", type=int, default=3000, help="seconds to wait (default 50 min)"
    )
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()

    if args.tag and PRERELEASE.search(args.tag):
        print(
            f"{args.tag} is a pre-release: Read the Docs does not build it, nothing to check"
        )
        return 0

    since = parse(args.since) - timedelta(seconds=60)
    version_url = API.format(project=args.project, version=args.version)
    deadline = time.monotonic() + args.timeout
    status = "no response yet"
    while time.monotonic() < deadline:
        try:
            version = get(version_url)
            builds = get(version_url + "builds/?limit=10")["results"]
        except urllib.error.HTTPError as error:
            status = f"API error: {error}"
            if error.code == 429:  # anonymous API calls are rate limited: back off
                delay = min(int(error.headers.get("Retry-After", 60)), 120)
                print(
                    f"  waiting: rate limited by Read the Docs, backing off {delay}s",
                    flush=True,
                )
                time.sleep(delay)
                continue
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as error:
            status = f"API error: {error}"
        else:
            if args.preflight:
                # `latest` tracks the branch: find the build of the release commit. A successful build
                # records its commit; a failed one may not, but then it is the first build that started
                # after the commit was pushed (builds come newest first)
                recent = [b for b in builds if parse(b["created"]) >= since]
                built = [
                    b for b in recent if b["success"] and b.get("commit") == args.sha
                ]
                first = recent[-1] if recent else None
                if built:
                    print(
                        f"OK: '{args.version}' built {args.sha} successfully: {built[0]['urls']['build']}"
                    )
                    return 0
                if first is None:
                    status = f"no build of '{args.version}' since the release commit was pushed"
                elif first["state"]["code"] != "finished":
                    status = f"build {first['id']} is {first['state']['name']}"
                elif not first["success"]:
                    print(
                        f"FAILED: the Read the Docs build of '{args.version}' for the release commit failed: "
                        f"{first['urls']['build']}"
                    )
                    print(
                        "The tag's docs would fail the same way and a published tag cannot move: fix it on "
                        "the default branch first, then tag the new commit."
                    )
                    return 1
                else:
                    status = f"'{args.version}' has not built {args.sha} yet"
            elif args.sha and version.get("identifier") != args.sha:
                status = f"'{args.version}' still points at {version.get('identifier')}, waiting for {args.sha} (tag not synced yet)"
            else:
                recent = [b for b in builds if parse(b["created"]) >= since]
                if not recent:
                    status = "no build of this version has started yet"
                elif recent[0]["state"]["code"] != "finished":
                    status = f"build {recent[0]['id']} is {recent[0]['state']['name']}"
                elif recent[0]["success"]:
                    print(
                        f"OK: build {recent[0]['id']} of '{args.version}' succeeded: {recent[0]['urls']['build']}"
                    )
                    return 0
                else:
                    print(
                        f"FAILED: Read the Docs build {recent[0]['id']} of '{args.version}' failed: {recent[0]['urls']['build']}"
                    )
                    print(
                        "The config is read from the tagged commit; fix it on master and release again (never move the tag)."
                    )
                    return 1
        print(f"  waiting: {status}", flush=True)
        time.sleep(args.interval)
    print(
        f"TIMEOUT after {args.timeout}s: {status}.\n"
        "Read the Docs can be slow to react to a new tag (it once took 20 minutes), so look at "
        f"https://app.readthedocs.org/projects/{args.project}/builds/ ; if nothing is queued there, "
        "check that the Read the Docs GitHub app is still connected to the repository."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
