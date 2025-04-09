# Apple_Leaves_Disease_Classification2
Apple_Leaves_Disease_Classification

Agriculture is considered one of the essential imapct on human lifes on Earth. Countires economy is mainly related to the agriculture. Crops is encounted with varoius pathological attacks. One example of such is apple trees which gets attacked with varoius diseases (e.g. apple scab, cedar apple
rust, or multiple diseases, etc) which influence thier production. This study presents an automatic apple leaves diseases detection using different deep learing approches based on the concept of transfer learning. The pre-trained models that have been used are densenet, effecientnet, MobileNetV2 and VGG16. Batch size,optimizer, and learning rate are the parameters which have been analysed in this work.
Commands:

sudo apt update

1269 python3 -m venv fl-env

1270 sudo apt update

1271 sudo apt install python3-pip python3-venv -y

1272 python3 -m venv fl-env

1273 source fl-env/bin/activate

1274 pip install --upgrade pip

1275 pip install flwr torch torchvision timm numpy

1276 mkdir -p ~/federated_learning/data/client1

1277 mkdir -p ~/federated_learning/data/client2

1278 mkdir -p ~/federated_learning/data/client3

1280 touch server.py client.py model.py

1313 python3 -m venv myenv

1314 source myenv/bin/activate

1315 python server.py

1316 python3 server.py

1317 pip install flwr

1320 pip install flower[superlink]

1324 deactivate

1325 python server.py

1328 pip install flwr torch torchvision timm numpy

1329 python3 -m venv fl-env

1330 python server.py

1331 python3 server.py

1332 sudo apt update && sudo apt upgrade -y

1333 sudo apt install python3-pip python3-venv git -y

1334 python3 -m venv fl-env

1335 source fl-env/bin/activate

1336 pip install --upgrade pip

1337 pip install torch torchvision timm flower

1339 python3 server.py

1340 which python

1341 which pip

1342 pip install flwr

1343 pip list | grep flwr

1345 pip install --upgrade pip

1348 python client.py client1

1349 print("Looking for data in:", dataset_path)

1350 print("Looking for data in:", data/client1)

1351 python3 client.py client1

1352 cat client.py

1353 pip install torch torchvision

1354 sudo apt install python3-venv

1355 python3 -m venv myenv

1356 source myenv/bin/activate

1357 pip install torch torchvision

1358 pip install --upgrade pip

1414 pip install scikit-learn

1418 rm -r clients/

1454 source fl-env/bin/activate

1461 rm -r clients/client1 clients/client2 clients/client3 val

1467 python client.py 3

1477 python client.py 1

1480 python client.py 2

1481 source fl-env/bin/activate

1484 pip install matplotlib

1485 python test.py /mnt/data/image.png

1493 python client.py 1

1495 python client.py 2

1497 python client.py 3

1498 source fl-env/bin/activate

1503 streamlit run app.py

1506 pip install streamlit

References

https://pytorch.org/

https://streamlit.io/

