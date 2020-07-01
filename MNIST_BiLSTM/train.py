
import tensorflow as tf
from misc import accuracy

def train_step(model, optimizer, images, label, train = True):

    with tf.GradientTape() as pred_tape:

        pred = model(images, train)

        pred_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits = pred)

    current_loss = tf.reduce_mean(pred_loss)
    accu = accuracy(pred, label)

    print("Current Training Loss : %f, Accuracy : %f" %(current_loss, accu))

    if train:
        gradient_of_predictor = pred_tape.gradient(pred_loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradient_of_predictor, model.trainable_variables))


def train(dataset, epochs, model, optimizer, train = True):

    for epoch in range(epochs):
        for batch in dataset:

            train_step(model, optimizer, batch[0], batch[1], train)


