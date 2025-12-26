도커(ROS2 Humble) 설치하는 법
==============================

도커 설치 및 실행법
-------------------------
```
sudo docker run -it --rm \
    --name=my_bot \
    --net=host \
    --ipc=host \
    --pid=host \
    --privileged \
    -v /dev:/dev \
    -v ~/ros2_ws:/root/ros2_ws \
    ros:humble-ros-base
```
위 명령어 실행 했는데
`Unable to find image 'ros:humble-ros-base' locall` 라는 오류가 뜨면
```
#ros:humble-ros-base 도커를 먼저 다운 받고
sudo docker pull ros:humble-ros-base
```
이거 실행하고 처음거 다시 실행
<img width="188" height="31" alt="image" src="https://github.com/user-attachments/assets/f42ce56a-1196-4f74-8fa5-7e362c1915b3" />
이렇게 바뀌면 성공    

도커 나가려면 
```
exit
```

<br /><br /><br />
도커 쓰기 전에 알아야 할 점
-------------------------
도커는 안에서 내용물 왔다리 갔다리 수정해도 껐다 키면 까먹음(단 ros2_ws에 따로 파일 만들거나 수정하는건 상관 없음 공유 폴더로 지정한거라)<br/>
그래서 만약에 pip3 install, apt install같은 거 한 후에 저장하고 싶으면    
**도커가 켜져있는 상태로**
다른 터미널에서 ssh접속해서 
```
sudo docker commit my_bot my_jetbot
```
이렇게 하면 기존 ros:humble-ros-base(얘는 인터넷에서 다운받은 도커)도커가  my_jetbot이라는 도커로 이름이 바뀌어 로컬 저장이 됨


<br /><br /><br />
도커 여는 단축키 지정
------------------------------------
도커 열 때 마다
```
sudo docker run -it --rm \
    --name=my_bot \
    --net=host \
    --ipc=host \
    --pid=host \
    --privileged \
    -v /dev:/dev \
    -v ~/ros2_ws:/root/ros2_ws \
    my_jetbot
```
이거 귀찮으니까
```
echo "alias start_jetbot='sudo docker run -it --rm  --name=my_bot --net=host   --ipc=host   --pid=host   --privileged  -v /dev:/dev -v ~/ros2_ws:/root/ros2_ws my_jetbot'" >> ~/.bashrc
source ~/.bashrc
```
이거 한 후 이제 `start_jetbot`칠 때마다 도커 실행됨



도커 만들고 설치해야할 것들
-------------------------------
**-도커안에서-**
```
apt update && apt install python3-pip -y
pip3 install numpy pyserial
apt install python3-opencv -y
apt install i2c-tools -y
pip3 install traitlets
pip3 install Adafruit-MotorHAT
```
```
cd /root/ros2_ws
mkdir src
cd src
git clone https://github.com/embedded-IEEE/jetank_move_test.git
```

아마 레포가 private이라 로그인 하라 뜰거임 일단 한번 로그인 하고 git clone 하고 
[#1] 
<br/><br/><br/>

```
cd /root/ros2_ws/src/jetank_move_test
. install.sh
```

**위에거 다 설치하고**<br/>
다른 터미널에서 ssh접속해서

```
sudo docker commit my_bot my_jetbot
```


servorinit실행
=============================
**-도커안에서-**
```
cd /root/ros2_ws/src/jetank_move_test/code
python3 servoInt.py
```
위에 파이썬 실행한 후에 바로 일반 컴퓨터 터미널에서
```
ros2 topic echo /robot_state
```
실행하면 토픽 publish/subscribe 확인 가능    

servotest실행
==================================
**도커안에서**
이 코드는 같은 인터넷 망에서 다른 애가 Publish한 걸<br/>
jetank가 subscribe해서 명령 받는 코드<br/><br/><br/>
```
python3 servotest_1.py
```

이후 다른 컴퓨터에서 
```
1.초기 위치
ros2 topic pub --once /jetank_cmd std_msgs/msg/String "data: 'home'"
2.  **팔 이동 (X=150, Y=50 좌표로 이동):**
ros2 topic pub --once /jetank_cmd std_msgs/msg/String "data: 'move 150 50'"
3.  **젯봇 전진:**
ros2 topic pub --once /jetank_cmd std_msgs/msg/String "data: 'forward'"
4.  **젯봇 정지:**
ros2 topic pub --once /jetank_cmd std_msgs/msg/String "data: 'stop'"
```
여기 명령어 보내면 움직임<br/>
<br/>
그리고 움직일 때 얘도 state publish하니까
다른 컴의 다른 터미널에서 
```
ros2 topic echo /robot_state
```

이거 하고 있으면 상태 pub할 수 있음




#### 만약에 안되면 여기 repo에 -> issue에 등록 부탁


