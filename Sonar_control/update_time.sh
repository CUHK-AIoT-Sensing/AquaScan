#!/bin/bash

service ntp stop

sudo ntpdate 192.168.1.28

date
