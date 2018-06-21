MLN_201806
==========

Public Repository of Machine Learning Nagoya 201806 (MLN_201806) Workshop.

With Docker
-----------

MLN\_201806 official Docker Image is available (Reuse of MLN\_201804,  https://hub.docker.com/r/nkats/mln_gym/ by [@n-kats](https://github.com/n-kats) ).  
Easy to use with the provided scripts.

```
$ docker pull nkats/mln_gym
$ docker.run.sh
(within the docker container)
# python handson_dqn.py
```

Without Docker
--------------

Install [OpenAI Gym](https://gym.openai.com/) framework and dependencies (See [https://github.com/openai/gym#installation](https://github.com/openai/gym#installation)) with Python3.x (recommended v3.6.x or higher). And then:

```
$ python3 handson_dqn.py
```
