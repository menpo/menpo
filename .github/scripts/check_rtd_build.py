#!/usr/bin/env python3
"""Wait for Read the Docs to build a released version and fail if the build fails.

Read the Docs reads its configuration (.readthedocs.yaml) from the tagged commit, so a tag with a
broken docs setup cannot be repaired afterwards (the tag must not move). This is run after a release
is published to make that visible straight away instead of leaving `stable` silently outdated.

Only the public API of the (public) project is used, no token is needed.
"""
import argparse
import json
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
    parser.add_argument("--project", default="menpo")
    parser.add_argument("--version", default="stable", help="Read the Docs version to check")
    parser.add_argument("--tag", help="release tag; pre-releases are skipped")
    parser.add_argument("--sha", help="commit the version must point at (the tagged commit)")
    parser.add_argument("--since", required=True, help="ISO time: only builds created after it count")
    parser.add_argument("--timeout", type=int, default=1200, help="seconds to wait (default 20 min)")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()

    if args.tag and PRERELEASE.search(args.tag):
        print(f"{args.tag} is a pre-release: Read the Docs does not build it, nothing to check")
        return 0

    since = parse(args.since) - timedelta(seconds=60)
    version_url = API.format(project=args.project, version=args.version)
    deadline = time.monotonic() + args.timeout
    status = "no response yet"
    while time.monotonic() < deadline:
        try:
            version = get(version_url)
            builds = get(version_url + "builds/?limit=5")["results"]
        except urllib.error.HTTPError as error:
            status = f"API error: {error}"
            if error.code == 429:  # anonymous API calls are rate limited: back off
                delay = min(int(error.headers.get("Retry-After", 60)), 120)
                print(f"  waiting: rate limited by Read the Docs, backing off {delay}s", flush=True)
                time.sleep(delay)
                continue
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as error:
            status = f"API error: {error}"
        else:
            if args.sha and version.get("identifier") != args.sha:
                status = f"'{args.version}' still points at {version.get('identifier')}, waiting for {args.sha} (tag not synced yet)"
            else:
                recent = [b for b in builds if parse(b["created"]) >= since]
                if not recent:
                    status = "no build of this version has started yet"
                elif recent[0]["state"]["code"] != "finished":
                    status = f"build {recent[0]['id']} is {recent[0]['state']['name']}"
                elif recent[0]["success"]:
                    print(f"OK: build {recent[0]['id']} of '{args.version}' succeeded: {recent[0]['urls']['build']}")
                    return 0
                else:
                    print(f"FAILED: Read the Docs build {recent[0]['id']} of '{args.version}' failed: {recent[0]['urls']['build']}")
                    print("The config is read from the tagged commit; fix it on master and release again (never move the tag).")
                    return 1
        print(f"  waiting: {status}", flush=True)
        time.sleep(args.interval)
    print(f"TIMEOUT after {args.timeout}s: {status}. Is the Read the Docs GitHub app still connected to the repository?")
    return 1


if __name__ == "__main__":
    sys.exit(main())
