#!/bin/bash

docker run -it --rm -v $(cd $(dirname $0) && pwd):/work nkats/mln_gym
