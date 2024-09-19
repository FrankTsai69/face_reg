#!/bin/bash
# (GPIO 6)按鈕按住3秒 自動執行 sudo poweroff ， 在 GPIO 19 的 LED 會亮起來
led=26
btn=23
cntMax=30
if [ -e /sys/class/gpio/gpiochip512 -a ! -e /sys/class/gpio/gpiochip0 ]
then
    (( btn +=512 ))
    (( led +=512 ))
elif [ ! -e /sys/class/gpio/gpiochip0 ]
then
  echo "檢查一下： /sys/class/gpio/gpiochipXXX"
  ls -l /sys/class/gpio/
  exit 99
fi

function use()
{
    [ ! -f /sys/class/gpio/gpio$1/direction ] && echo $1 > /sys/class/gpio/export
}

function out()
{
    [ -f /sys/class/gpio/gpio$1/direction ] && echo out > /sys/class/gpio/gpio$1/direction

}

function on()
{
    [ -f /sys/class/gpio/gpio$1/value ] && echo 1 > /sys/class/gpio/gpio$1/value
}

function off()
{
    [ -f /sys/class/gpio/gpio$1/value ] && echo 0 > /sys/class/gpio/gpio$1/value
}

function switch()
{
    [ -f /sys/class/gpio/gpio$1/value ] && echo $2 > /sys/class/gpio/gpio$1/value
}

function get()
{
    [ -f /sys/class/gpio/gpio$1/value ] && cat /sys/class/gpio/gpio$1/value
}

function check()
{
    [ -f /sys/class/gpio/gpio$1/direction ] && cat /sys/class/gpio/gpio$1/direction
}

use $btn
use $led
sleep 0.5
out $led
count=0
for ((;;))
do 
            sleep 0.1
            press=$(get $btn)
	    ledout=$(check $led)
	    if ((ledout=="in"));then
		out $led
	    fi
            if [ -z "$press"  ] ; then
                 echo  gpio $btn  is not  ready
                 break
            fi
            if ((press==1)) ; then
		  on $led
                  if  (( ++count  >= $cntMax )) ; then
                     count=0
                     off $led
		     lxterminal -e /home/12/face_reg/auto_script/auto.sh
                  fi
            else 
                     count=0
		     off $led
            fi
done;

