# EdgeVision TensorFlow Notes — Part 1: Foundations
### Your project: Wafer Defect Detection using MobileNetV3Small
### Classes: bridge, clean, cmp, crack, ler, open, others, vias

> 70% theory, 30% code. Every topic is explained through your actual project decisions.
> Format per topic: What it is → How it works → In Your Project → What You Missed → Interview Q&A

---

## Topic 1 — TensorFlow Basics: Tensors, Ranks, Shapes, Eager Execution

### What it is

TensorFlow is a machine learning framework built by Google. At its core, everything in TensorFlow is a **tensor** — a multi-dimensional array of numbers. Every image you feed into your model, every weight inside MobileNetV3Small, every prediction your model outputs — all of it is a tensor. When you write `image / 255.0` in your preprocess function, you are performing a tensor operation. When your model processes a batch of 32 wafer images, those images travel through the network as a 4D tensor of shape `(32, 224, 224, 3)`.

### How it works

A tensor is defined by three things: its **shape**, its **rank**, and its **dtype**.

**Rank** is the number of dimensions. Think of it like this — a single number lives in 0 dimensions, a list lives in 1, a table lives in 2, and an image lives in 3.

- Rank 0 (scalar): a single confidence score — `tf.constant(0.91)`
- Rank 1 (vector): class probabilities — `tf.constant([0.1, 0.05, 0.6, 0.25, ...])`
- Rank 2 (matrix): a single grayscale image — shape `(224, 224)`
- Rank 3 (3D tensor): one RGB image — shape `(224, 224, 3)`
- Rank 4 (4D tensor): a batch of RGB images — shape `(32, 224, 224, 3)`

**Shape** tells you exactly how many elements exist along each dimension. When your training loop runs, data moves as `(32, 224, 224, 3)` — 32 images, 224 height, 224 width, 3 channels.

**Dtype** is the data type. Images typically start as `uint8` (integers 0–255). After normalization they become `float32` (decimals 0.0–1.0). Weights inside the model are `float32` by default. When you do TFLite float16 conversion, you're changing the dtype of the weights to use 16 bits instead of 32 — cutting size roughly in half.

**tf.constant vs tf.Variable** — these are the two ways to create tensors.

`tf.constant` creates a tensor whose value is fixed and never changes. Use it for input data, fixed configurations, anything that does not get updated during training.

`tf.Variable` creates a tensor that can be updated. Every single weight inside your MobileNetV3Small — the 2.5 million parameters — is a `tf.Variable`. Gradient descent computes how much each weight should change and updates them in place. When you call `base_model.trainable = False`, you are telling TensorFlow to stop computing gradients for those variables during backpropagation — they stay frozen.

**Eager execution** means TensorFlow runs operations immediately and returns real values, like normal Python. Before TensorFlow 2.x, you had to write the full computation graph first and then run it in a `Session.run()` call — you could not print intermediate values or debug easily. TF 2.x made eager execution the default, so you can print tensors, inspect shapes mid-code, and it behaves like NumPy. This is why modern TF code looks clean and readable.

### In Your EdgeVision Project

```python
import tensorflow as tf
tf.keras.backend.clear_session()
```

You called this at the very top of every notebook. This resets the entire Keras global state — it clears all model variables, resets layer name counters, and frees GPU memory from any previous run in the same Colab session. Without it, if you rerun a cell that defines a model, TensorFlow starts naming layers `dense_1`, `dense_2`, `dense_3` instead of `dense` — because it thinks those names are already taken. Over multiple reruns, you can also accumulate old model weights in GPU memory and eventually crash. `clear_session()` wipes the slate clean.

When you wrote `image / 255.0` in your preprocess function, TensorFlow performed that division element-wise across an entire batch tensor of shape `(32, 224, 224, 1)` — all 32 images, every pixel, in one vectorized C++ operation. No Python loop. This is the core performance advantage of tensor operations over plain Python.

### What You Missed

You never directly used `tf.constant` or `tf.Variable` explicitly — you always went through Keras layers, which handle them internally. This is fine for building models, but it means you don't have a clear mental model of what "freezing layers" actually does at the tensor level, or why `tf.GradientTape` watches specific variables. Understanding tensors directly makes Grad-CAM, quantization, and fine-tuning much more intuitive.

### Interview Q&A

**Q: What is the difference between tf.constant and tf.Variable?**
`tf.constant` is immutable — its value is fixed after creation, used for input data. `tf.Variable` is mutable and participates in gradient computation — used for model weights that get updated during backpropagation via gradient descent.

**Q: What does eager execution mean in TensorFlow 2.x, and why does it matter?**
Operations execute immediately and return concrete tensor values, just like NumPy. In TF 1.x you had to build a static computation graph first and run it in a Session. Eager execution makes debugging far easier because you can print and inspect tensors at any point in your code.

**Q: Why do you call clear_session() at the start of a Colab notebook?**
It resets Keras's global state — clears all model variables, layer name counters, and frees GPU memory from previous runs. Without it, rerunning training cells in the same session causes naming conflicts, memory buildup, and unpredictable behavior.

---

## Topic 2 — Loading Data with tf.data and image_dataset_from_directory

### What it is

`tf.data` is TensorFlow's data pipeline API. It handles loading, batching, shuffling, and preprocessing datasets efficiently. `image_dataset_from_directory` is a Keras utility that wraps `tf.data` — it reads images from a folder structure where each subfolder is a class, creates a `tf.data.Dataset` object, and handles batching and label assignment automatically.

### How it works

When you call `image_dataset_from_directory`, it does several things in order. First it scans the directory and maps each subfolder name to an integer label — alphabetically. So `bridge → 0`, `clean → 1`, `cmp → 2`, `crack → 3`, `ler → 4`, `open → 5`, `others → 6`, `vias → 7`. This is why `class_names = train_data.class_names` gives you that ordered list — the labels in your dataset correspond to those indices.

Then it creates a `tf.data.Dataset` that reads images lazily — it does not load all images into RAM at once. It loads them in batches, on demand, as training requests them. This is critical for large datasets that would otherwise not fit in memory.

**batch_size=32** means the dataset yields 32 images at a time. Each call to the model during training sees 32 images, computes the loss across all 32, and updates weights based on that average gradient. Larger batch sizes give more stable gradients but require more memory. 32 is a common default that works well on Colab's free GPU.

**shuffle=True** randomizes the order in which images are served. This matters because if you feed all "bridge" images first, then all "clean" images, the model sees one class at a time and the gradients become biased toward whatever class it saw most recently. Shuffling ensures each batch is a random mix of classes.

**image_size=(224, 224)** resizes every image to 224×224 pixels before yielding them. This is necessary because neural networks require fixed-size inputs. MobileNetV3Small was designed for 224×224 input.

**What you get back** is a `tf.data.Dataset` object — not a list, not a NumPy array. It is a lazy generator that produces `(images, labels)` tuples. When you loop `for images, labels in train_data:`, you are consuming these batches one at a time.

**What .cache() does** — the first time the dataset is iterated, it reads images from disk and stores them in memory (or on disk if memory is limited). On the second epoch, instead of reading from disk again, it reads from the cache. Disk I/O is usually the training bottleneck in Colab, not GPU computation. Without `.cache()`, your GPU sits idle waiting for data to load from Drive between batches.

**What .prefetch() does** — while the GPU is training on batch N, `.prefetch()` tells the CPU to start loading batch N+1 in the background. Without it, training is sequential: load batch → train → load next batch → train. With it, loading and training happen in parallel — the pipeline never starves the GPU.

Think of it like a factory assembly line. Without prefetch, the machine stops to wait for raw materials. With prefetch, the next batch of materials arrives before the machine finishes the current one.

### In Your EdgeVision Project

```python
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=(224, 224),
    color_mode="grayscale",
    batch_size=32,
    shuffle=True
)
class_names = train_data.class_names
print(class_names)
```

You used `color_mode="grayscale"` because wafer defect images are grayscale — they contain no color information, just intensity values. This makes each image shape `(224, 224, 1)` instead of `(224, 224, 3)`. The `1` is a single channel.

Then in your preprocess function you expanded it back to 3 channels using `tf.repeat(image, 3, axis=-1)`. This was necessary because MobileNetV3Small expects 3-channel RGB input — it was trained on color ImageNet images and its first layer is designed for 3 channels. Repeating the grayscale channel three times is the standard workaround.

### What You Missed

You did not use `.cache()` or `.prefetch()` anywhere. This is the single biggest performance gap in your training pipeline. In Colab, images are stored on Google Drive and read over a network connection. Without caching, every epoch re-reads every image from Drive. This explains why your training was slow. The fix is two lines:

```python
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_data  = test_data.cache().prefetch(buffer_size=AUTOTUNE)
```

`AUTOTUNE` tells TensorFlow to automatically decide how many batches to prefetch based on available resources. Adding these two lines to your pipeline would meaningfully reduce training time per epoch.

### Interview Q&A

**Q: What does image_dataset_from_directory return, and how does it assign labels?**
It returns a `tf.data.Dataset` that yields `(images, labels)` batches. Labels are assigned as integers corresponding to subfolder names sorted alphabetically — so the first folder alphabetically gets label 0, the second gets label 1, and so on. `dataset.class_names` gives you the ordered list.

**Q: What is the difference between .cache() and .prefetch() in a tf.data pipeline?**
`.cache()` stores the dataset in memory after the first epoch so disk reads only happen once. `.prefetch()` overlaps data loading and model training — while the GPU trains on batch N, the CPU loads batch N+1 in the background. Both together eliminate the data loading bottleneck that otherwise idles the GPU.

**Q: Why is shuffle=True important during training but not necessarily during evaluation?**
Shuffling ensures each batch contains a random mix of classes, preventing the optimizer from seeing biased gradients. During evaluation you don't update weights, so order doesn't affect results — but shuffling the test set can make debugging harder because predictions are no longer in a predictable order.

---

## Topic 3 — Preprocessing and Augmentation

### What it is

Preprocessing transforms raw image data into the format a neural network expects — normalized pixel values, correct shape, and correct dtype. Augmentation creates artificial variations of training images — flips, rotations, zooms — to simulate a wider variety of data without collecting more images. Both happen in the data pipeline before images reach the model.

### How it works

**Normalization** converts pixel values from the range `[0, 255]` (integer) to `[0.0, 1.0]` (float). Neural networks use gradient descent to update weights, and gradient descent works best when inputs are small, consistent numbers centered around 0. If you feed raw pixel values like 200 or 150, the gradients become very large and the optimizer struggles to converge. Dividing by 255.0 keeps everything in `[0, 1]` and makes training stable.

Some pretrained models expect inputs in `[-1, 1]` instead of `[0, 1]`. MobileNetV3Small's preprocessing function scales to `[-1, 1]` internally — which means if you use `tf.keras.applications.mobilenet_v3.preprocess_input()`, you should not also divide by 255.0 yourself. You'd be double-normalizing.

**Why tf.repeat instead of cv2.cvtColor** — `tf.repeat(image, 3, axis=-1)` duplicates the single grayscale channel three times along the last axis, turning `(224, 224, 1)` into `(224, 224, 3)`. This runs inside the `tf.data` pipeline as a TensorFlow operation, which means it runs on GPU and is part of the graph. `cv2.cvtColor` is a CPU-only NumPy operation — you cannot use it inside `.map()` on a tf.data pipeline because the pipeline expects TensorFlow operations. This is exactly why you used `tf.repeat`.

**Data Augmentation** applies random transformations to training images each epoch. The key word is **random** — each time the same image is seen, it looks slightly different. This prevents the model from memorizing exact pixel patterns and forces it to learn the underlying structure of defects. Common augmentations for wafer images: random horizontal flip (defects appear from any direction), random rotation (wafer orientation varies), random zoom (defect scale varies).

Augmentation only applies during training, not during validation or inference. The model should see clean, unmodified images when being evaluated.

There are two ways to do augmentation in TensorFlow. You can add augmentation layers inside the model itself, or you can apply them in the `.map()` step of your tf.data pipeline. The model-layer approach is cleaner because the augmentation is tied to the model and automatically disabled during `model.predict()`.

```python
# Augmentation as model layers (the right way)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
```

These layers are aware of whether the model is in training mode or inference mode. During `model.fit()`, they apply random transforms. During `model.predict()` or `model.evaluate()`, they pass the image through unchanged.

### In Your EdgeVision Project

```python
def preprocess(image, label):
    image = image / 255.0
    image = tf.repeat(image, 3, axis=-1)
    return image, label

train_data = train_data.map(preprocess)
test_data  = test_data.map(preprocess)
```

Your preprocess function does two things: normalizes to `[0, 1]` and converts grayscale to pseudo-RGB. You applied the same preprocess function to both train and test data, which is correct — normalization should be consistent.

The `.map()` call applies this function to every `(image, label)` pair in the dataset lazily. It doesn't run immediately — it adds the transformation to the pipeline so it executes when data is actually consumed during training.

### What You Missed

You did not do any data augmentation. Your model trained on the same 224×224 pixels every epoch. Wafer defect datasets are typically small and imbalanced — your 8 classes (bridge, clean, cmp, crack, ler, open, others, vias) likely have very different image counts. Without augmentation, minority classes like `cmp` or `ler` are seen far fewer times than `clean`, making it harder for the model to learn their features.

Adding random flips and small rotations would have been especially helpful because wafer defects don't have a fixed orientation — a crack can appear at any angle on the wafer.

Also, you should be using MobileNetV3's own preprocessing function instead of manual `/ 255.0`:

```python
# More correct for MobileNetV3
image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
```

This scales to `[-1, 1]` which is what MobileNetV3 expects based on how it was trained on ImageNet.

### Interview Q&A

**Q: Why do we normalize image pixel values before feeding them to a neural network?**
Neural networks train via gradient descent, which converges faster and more stably when inputs are small values close to 0. Raw pixel values in [0, 255] produce large gradients that make optimization unstable. Normalizing to [0, 1] or [-1, 1] keeps gradient magnitudes manageable.

**Q: Why does data augmentation only apply during training and not during inference?**
Augmentation introduces random artificial variations to prevent overfitting — it's a regularization technique. During inference, you want the model to make predictions on the actual image, not a randomly flipped or rotated version of it. Keras augmentation layers automatically detect training vs inference mode and behave accordingly.

**Q: What is the difference between applying augmentation in the tf.data pipeline vs as layers inside the model?**
In the tf.data pipeline, augmentation runs on CPU and is separate from the model — you have to manually skip it during evaluation. As model layers, augmentation is part of the model graph and automatically disabled during `model.evaluate()` and `model.predict()`. The model-layer approach is cleaner and less error-prone.

---

## Topic 4 — CNN Fundamentals

### What it is

A Convolutional Neural Network (CNN) is a type of neural network designed specifically for image data. Instead of looking at every pixel independently like a regular dense network would, a CNN uses small filters that slide across the image and detect local patterns — edges, textures, shapes — that are the same regardless of where they appear in the image. This is how MobileNetV3Small understands wafer defects.

### How it works

**Convolution and filters** — imagine you have a 3×3 filter (also called a kernel). This filter slides across your 224×224 image, one position at a time, and at each position it does a dot product between its 9 values and the 9 pixels underneath it. The result is a single number. After sliding across the entire image, you get a 2D output called a **feature map**. Each filter detects one specific pattern. A filter with high values on one side and low on the other detects edges. A filter with a cross pattern detects corners. The model learns what values these filters should have during training — you don't define them manually.

A `Conv2D(32, (3,3))` layer means 32 different 3×3 filters — so 32 different patterns being detected simultaneously, producing 32 feature maps. Early layers learn low-level patterns (edges, blobs). Deeper layers combine those into high-level patterns (shapes of defects like cracks or bridges).

**Padding** — when a 3×3 filter is at the edge of the image, it doesn't have enough neighbors. `padding='same'` adds a border of zeros around the image so the filter can slide fully and the output feature map is the same size as the input. `padding='valid'` means no padding — the output shrinks slightly.

**Stride** — how many pixels the filter moves each step. Stride 1 means move one pixel at a time (fine-grained). Stride 2 means jump two pixels (the output is half the size). Larger strides reduce spatial dimensions.

**Pooling** — after convolution, you often apply pooling to reduce the spatial size of feature maps. `MaxPooling2D(2,2)` takes every 2×2 block and keeps only the maximum value. This makes the representation smaller (faster to process) and slightly position-invariant — the exact pixel location of a feature matters less.

**GlobalAveragePooling2D** — instead of keeping spatial dimensions, this takes each feature map and averages all its values into a single number. If your model has 576 feature maps at the end of MobileNetV3's convolutional layers, GlobalAveragePooling2D collapses them into a vector of 576 numbers. This is much better than `Flatten` for transfer learning because it dramatically reduces parameters and is more spatially invariant.

**BatchNormalization** — after a convolution or dense layer, the activations can have very different scales. Batch normalization normalizes them to have mean 0 and variance 1 across the batch, then applies learnable scale and shift parameters. This stabilizes training significantly, allows higher learning rates, and acts as mild regularization. You used it right after GlobalAveragePooling2D in your model.

**Dropout** — randomly sets a fraction of neuron outputs to zero during each training step. `Dropout(0.4)` means 40% of neurons are randomly silenced every forward pass. This prevents the network from relying too heavily on any single neuron and forces redundancy — a form of regularization that reduces overfitting. Dropout is disabled during inference automatically by Keras.

**Dense layers** — fully connected layers at the end of the network. After global pooling gives you a vector of features, a Dense(128) layer learns a weighted combination of those 576 features into 128 outputs. The final Dense(8) layer (8 = number of classes) produces the raw scores for each class.

**Softmax** — the activation on the final Dense(8) layer. It converts raw scores into probabilities that sum to 1.0. `[0.02, 0.85, 0.01, 0.03, 0.01, 0.04, 0.01, 0.03]` — this means the model is 85% confident this is class 1 (clean).

### In Your EdgeVision Project

Your model's head (the part you built on top of MobileNetV3Small):

```python
model = tf.keras.Sequential([
    base_model,                                    # MobileNetV3Small frozen
    tf.keras.layers.GlobalAveragePooling2D(),      # (None, 576) — collapses spatial dims
    tf.keras.layers.BatchNormalization(),           # stabilizes the 576 features
    tf.keras.layers.Dense(128, activation="relu"), # learns wafer-specific combinations
    tf.keras.layers.Dropout(0.4),                  # regularization
    tf.keras.layers.Dense(8, activation="softmax") # 8 class probabilities
])
```

MobileNetV3Small outputs a tensor of shape `(batch, 7, 7, 576)` from its last convolutional block — 576 feature maps, each 7×7 pixels. GlobalAveragePooling2D collapses that to `(batch, 576)`. Then your Dense head learns to map those 576 features to 8 defect classes.

The reason your val_acc (79%) was higher than train_acc (69%) is directly related to Dropout and BatchNormalization. Both behave differently during training vs evaluation. During training, Dropout randomly drops 40% of neurons — making it harder for the model to perform well on training data. During evaluation, Dropout is disabled — all neurons are active. BatchNormalization uses batch statistics during training (which are noisy) but uses the learned running averages during inference (which are more stable). Together, these make validation metrics look better than training metrics, especially early in training.

### What You Missed

You used MobileNetV3Small as a black box. Knowing its internal architecture matters for interviews. MobileNetV3 uses **inverted residuals with linear bottlenecks** (from MobileNetV2) and **squeeze-and-excitation blocks** that learn to weight which feature maps are most important. The "V3Small" variant is the lightweight version designed for edge devices — it has fewer parameters than MobileNetV3Large and MobileNetV2 while being faster.

You also never used `Conv2D` directly in your project — which means you don't have hands-on intuition for filter counts, kernel sizes, and strides. If asked to build a CNN from scratch in an interview, knowing how these pieces connect matters.

### Interview Q&A

**Q: Why is GlobalAveragePooling2D preferred over Flatten in transfer learning?**
Flatten preserves all spatial positions and creates a very large vector — leading to millions of parameters in the next Dense layer. GlobalAveragePooling2D averages each feature map into a single number, creating a compact vector that is also spatially invariant. It reduces parameters significantly and generalizes better to new domains.

**Q: Why was your validation accuracy higher than training accuracy in EdgeVision?**
Because Dropout and BatchNormalization behave differently in training mode vs inference mode. During training, Dropout randomly silences 40% of neurons and BatchNormalization uses noisy batch statistics — both add noise that reduces apparent training accuracy. During validation, Dropout is disabled and BatchNormalization uses stable running averages, so the model performs better.

**Q: What does BatchNormalization actually do and why does it help training?**
It normalizes the outputs of a layer to have zero mean and unit variance across the batch, then applies learnable scale and shift. This prevents activations from becoming too large or too small (vanishing/exploding gradients), allows the model to use higher learning rates, and speeds up convergence.

---

## Topic 5 — Building Models: Sequential, Functional, and Subclassing

### What it is

Keras offers three ways to define a model. Sequential API stacks layers linearly — input flows through each layer one after another. Functional API lets you define arbitrary graph structures — multiple inputs, skip connections, branching. Model subclassing lets you write a model as a Python class with a `call()` method for maximum flexibility. You used Sequential. Most real projects use Functional.

### How it works

**Sequential API** is the simplest. Layers are added in order and the output of one becomes the input of the next. It works perfectly when your architecture is a straight line. The limitation is that you cannot create branches, skip connections, or multiple inputs/outputs. You also cannot easily access intermediate layer outputs — which matters for Grad-CAM.

```python
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(8, activation="softmax")
])
```

**Functional API** defines layers as functions applied to tensor inputs. You explicitly pass tensors through each layer, which means you can branch, merge, and access any intermediate output. It's the standard for anything beyond a simple stack.

```python
inputs  = tf.keras.Input(shape=(224, 224, 3))
x       = base_model(inputs, training=False)
x       = tf.keras.layers.GlobalAveragePooling2D()(x)
x       = tf.keras.layers.BatchNormalization()(x)
x       = tf.keras.layers.Dense(128, activation="relu")(x)
x       = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(8, activation="softmax")(x)
model   = tf.keras.Model(inputs, outputs)
```

This defines exactly the same architecture as your Sequential model, but now you have explicit references to `inputs` and every intermediate tensor `x`. This is important for Grad-CAM because you need to create a sub-model that outputs at a specific intermediate layer.

**Why Functional would've been better for your Grad-CAM** — in your Pipeline.ipynb, you had to reach inside the Sequential model to extract the backbone with `model.layers[0]` and then build a separate `conv_model` from the backbone's input and a specific conv layer's output. This was awkward and fragile. With Functional API, you could directly create `grad_model = tf.keras.Model(inputs, [conv_layer.output, model.output])` cleanly.

**Model subclassing** — you write a class that inherits from `tf.keras.Model` and define `__init__` (where you define layers) and `call()` (where you define the forward pass). This is the most flexible approach and used in research code, but it's harder to use `model.summary()` or `model.save()` reliably. You don't need this for your project, but knowing it exists and when it's used matters.

```python
class WaferClassifier(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.base      = tf.keras.applications.MobileNetV3Small(include_top=False, weights="imagenet")
        self.pool      = tf.keras.layers.GlobalAveragePooling2D()
        self.bn        = tf.keras.layers.BatchNormalization()
        self.dense     = tf.keras.layers.Dense(128, activation="relu")
        self.dropout   = tf.keras.layers.Dropout(0.4)
        self.out       = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, x, training=False):
        x = self.base(x, training=training)
        x = self.pool(x)
        x = self.bn(x, training=training)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        return self.out(x)
```

Notice you pass `training=training` explicitly to layers like BatchNorm and Dropout — this is important in subclassing because Keras doesn't automatically propagate training mode through custom `call()` methods.

**model.summary()** gives you the layer names, output shapes, and parameter counts. You used this to verify your architecture. The total parameters in your model is MobileNetV3Small's 2.5M + your head's ~75K. When you freeze the base, only the 75K parameters are trainable during phase 1.

### What You Missed

You should rebuild your model using the Functional API. It would make your Grad-CAM code cleaner, give you explicit access to layer outputs, and is what you'll see in every real-world codebase and interview question. Sequential is fine for prototyping, but Functional is the professional standard.

### Interview Q&A

**Q: What is the difference between Sequential and Functional API in Keras?**
Sequential stacks layers linearly with no branches — simple and clean but inflexible. Functional API lets you define arbitrary computation graphs with explicit tensor connections — supports multiple inputs/outputs, skip connections, and access to intermediate layers. Functional is the standard for real projects.

**Q: When would you use Model subclassing over Functional API?**
When you need custom forward pass logic that can't be expressed as a static graph — dynamic architectures, custom training steps, or research experiments where the computation changes based on input at runtime. For production and standard architectures, Functional API is preferred because it supports model.save(), model.summary(), and visualization better.

**Q: How would you access an intermediate layer's output in a Keras model?**
With Functional API, you create a new model: `sub_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('layer_name').output)`. With Sequential, you have to wrap or restructure because there's no direct tensor reference to intermediate outputs.

---

*End of Part 1 — Foundations*
*Part 2 covers: model.compile(), training loop, evaluation, callbacks, transfer learning, Grad-CAM*
