#!/bin/bash -v
git pull origin feature/dev:master
chmod +x $0
git add $0
git commit -m 'chmod'
git push origin master:feature/dev