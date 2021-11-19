#!/bin/bash -v
git pull origin feature/dev:master
chmod +x "$1"
git add "$1"
git commit -m 'chmod'
git push origin master:feature/dev