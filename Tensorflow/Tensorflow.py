# TensorFlow 를 사용하기 위해 import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model
x = tf.placeholder(tf.float32, [None, 784]) # MNIST 이미지들의 어떤 숫자들이든 입력할 수 있기를 원하는데, 각 이미지들은 784차원의 벡터로 단조화되어 있음. 우리는 이걸 [None, 784] 형태의 부정소숫점으로 이루어진 2차원 텐서로 표현(None은 해당 값의 어떤 길이도 될 수 있음)
W = tf.Variable(tf.zeros([784, 10])) # 784차원의 이미지 벡터를 곱하여 10차원 벡터의 증거를 만듬
b = tf.Variable(tf.zeros([10])) #  bb는 [10]의 형태이므로 출력에 더함
y = tf.nn.softmax(tf.matmul(x, W) + b) # 첫번째로, 우리는 tf.matmul(x, W) 표현식으로 x 와 W를 곱합니다. 이 값은 Wx가 있던 우리 식에서 곱한 결과에서 뒤집혀 있는데, 이것은 x가 여러 입력으로 구성된 2D 텐서일 경우를 다룰 수 있게 하기 위한 잔재주입니다. 그 다음 b를 더하고, 마지막으로 tf.nn.softmax 를 적용.

y_ = tf.placeholder(tf.float32, [None, 10]) # 교차 엔트로피를 구현하기 위해 우리는 우선적으로 정답을 입력하기 위한 새 placeholder를 추가

cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # 첫번째로, tf.log는 y의 각 원소의 로그값을 계산합니다. 그 다음, y_ 의 각 원소들에, 각각에 해당되는 tf.log(y)를 곱합니다. 마지막으로, tf.reduce_sum은 텐서의 모든 원소를 더함.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # 이 경우 TensorFlow에게 학습도를 0.01로 준 경사 하강법(gradient descent) 알고리즘을 이용하여 교차 엔트로피를 최소화하도록 명령했습니다. 경사 하강법은 TensorFlow 가 각각의 변수들을 비용을 줄이는 방향으로 약간씩 바꾸는 간단한 방법

# Session
init = tf.initialize_all_variables() # 실행 전 마지막으로 우리가 만든 변수들을 초기화하는 작업을 추가

# 세션에서 모델을 시작하고 변수들을 초기화하는 작업을 실행
sess = tf.Session()
sess.run(init)

# Learning
# 각 반복 단계마다, 학습 세트로부터 100개의 무작위 데이터들의 일괄 처리(batch)들을 가져옵니다. placeholders를 대체하기 위한 일괄 처리 데이터에 train_step 피딩을 실행
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step,

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # tf.argmax는 특정한 축을 따라 가장 큰 원소의 색인을 알려주는 엄청나게 유용한 함수, tf.equal 을 이용해 예측이 실제와 맞았는지 확인할 수 있습니다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #  테스트 데이터를 대상으로 정확도를 확인

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
