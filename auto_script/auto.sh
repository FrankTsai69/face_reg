#! /bin/bash
cd /home/12/face_reg
export DISPLAY=:0
/home/12/py_envs/bin/python3 /home/12/face_reg/demo.py && (echo "excete" >> /home/12/face_reg/auto_script/log.txt) || (echo "false" >> /home/12/face_reg/auto_script/log.txt)

date>> /home/12/face_reg/auto_script/log.txt
echo "" >>/home/12/face_reg/auto_script/log.txt