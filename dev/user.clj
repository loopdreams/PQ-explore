(ns user
  (:require [libpython-clj2.python :as py]))


(py/initialize! :python-executable (str (System/getenv "HOME") "/.pyenv/versions/3.12.1/envs/fresh-venv/bin/python3.12")
                :library-path (str (System/getenv "HOME") "/.pyenv/versions/3.12.1/lib/python3.12/site-packages/"))
