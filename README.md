# DCN 
This is a simplified demo of the paper "Deep Learning based Densely Connected  Network for Load Forecasting".

## Environment requirement
This code is implemented on the Ubuntu 17 system using Python2. If you want to run it with Python3. You should bridge the grammar gap between Python2 and Python3 by yourself. For example, the print in Python2 is
```
print 'hello world'
```
If in Python3, it should be
```
print('hello world')
```
In addition, the required libraries should be installed, such as tensorflow, keras, numpy, opencv, matplotlib, sklearn, scipy. You can use the following comand to install it in the Linux system.
```
sudo pip install "library name"
```
## Running steps
First, you should run the file "build.sh" to prepare the running environment. you can use the following commands.
```
sudo chmod +x build.sh
./build.sh
```
Then, for the deterministic forecasting, you can use the following comand to train the model.
```
python train.py
```
Next, you can validate the trained model using the following command.
```
python validation.py
```
Furthermore, for training the interval forecasting model, you should run the following two commands, respectively.
```
python up_train.py
python down_train.py
```
And you can use this command to verify its performance.
```
python pro_validation.py
```

