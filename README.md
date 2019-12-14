# PixelCNN++

TensorFlow 2 distributed implementation of PixelCNN++ [[1]](https://arxiv.org/abs/1701.05517).

## Setup

To install the required dependencies you can simply run:
```
pip install requirements.txt
```

You will need a machine that preferablly has multiple GPU's. Training PixelCNN++ on modest datasets like Cifar10 can take days or weeks with multiple GPU's to obtains results comparable to the original paper.

## Training

To train on a single GPU or CPU:
```
python main.py --config experiments/mnist.gin single
```

To train with multiple GPUs:
```
python main.py --config experiments/mnist.gin multigpu
```

You can change Gin parameters on the command line by using the `--binding` flag. You can use this flag multiple times. For example,
```
python main.py --config experiments/mnist.gin --binding "train.batch_size=128" multigpu
```

## Custom Datasets

You can easily use a custom dataset with this implementation.
You just need to define a function that returns a tuple `(train, test)` which returns a train and test Tensorflow `tf.data.Dataset` object. You can then bind `train.dataset_fn` to your new dataset function using Gin.

Note you can use a dataset with any number of channels without any further modification. If you use one channel make sure that the image shape is still `(W, H, 1)`.

## Omissions

* Currently, evaluation doesn't employ Polyak averaging over previous weights. This is planned with the Tensorflow Addons `tfa.optimizers.MovingAverage`.

* I don't currently use the "autoregressive channel" discussed in the original paper.

* WeightNormalization doesn't use data dependent initialization. This is planned in the future with Tensorflow Addons.


## References

[1] [PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications](https://arxiv.org/abs/1701.05517)


[2] [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)


[3] [OpenAI PixelCNN++ Implementation](https://github.com/openai/pixel-cnn)
