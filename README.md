# Online learning

## iCaRL

Reference from [here](https://github.com/DRSAD/iCaRL).

- When adding new category, create new node in FC layer.

```python
def Incremental_learning(self, numclass):
    weight = self.fc.weight.data
    bias = self.fc.bias.data
    in_feature = self.fc.in_features
    out_feature = self.fc.out_features
    self.fc = nn.Linear(in_feature, numclass, bias=True)
    self.fc.weight.data[:out_feature] = weight
    self.fc.bias.data[:out_feature] = bias
```

Representation Learning

- The loss is combined by a standard classification loss and a distillation loss.

```python
def _compute_loss(self, indexs, imgs, target):
    # target is the ground truth
    output = self.model(imgs)
    target = get_one_hot(target, self.numclass)
    output, target = output.to(device), target.to(device)
    if self.old_model == None:
        return F.binary_cross_entropy_with_logits(output, target)
    else:
        old_target = torch.sigmoid(self.old_model(imgs))
        old_task_size = old_target.shape[1]
        target[..., :old_task_size] = old_target
        return F.binary_cross_entropy_with_logits(output, target)
```

- Nearest-Mean-of-Exemplars Classification

```python
def classify(self, test):
    # test is the images for test.
    result = []
    test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
    # test = self.model.feature_extractor(test).detach().cpu().numpy()
    class_mean_set = np.array(self.class_mean_set)
    for target in test:
        x = target - class_mean_set
        x = np.linalg.norm(x, ord=2, axis=1)
        x = np.argmin(x)
        result.append(x)
    return torch.tensor(result)
```

Exemplar Management

- Reduce exemplar set

```python
def _reduce_exemplar_sets(self, m):
    # m=int(self.memory_size/self.numclass)
    # m is the number of exemplar in each category.
    for index in range(len(self.exemplar_set)):
        self.exemplar_set[index] = self.exemplar_set[index][:m]
        # self.exemplar_set is a list, every elements in this list is a list.
        print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))


def _construct_exemplar_set(self, images, m):
    class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
    # `images` are the images belong to a specific category
    # `self.compute_class_mean` get the average of normalized feature map of `images`
    exemplar = []
    now_class_mean = np.zeros((1, 512))

    for i in range(m):
        # shape：batch_size*512
        x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
        # the detailed explaination can be find in Algorithm 4 in the original paper.
        # shape：batch_size
        x = np.linalg.norm(x, axis=1)
        index = np.argmin(x)
        now_class_mean += feature_extractor_output[index]
        exemplar.append(images[index])

    print("the size of exemplar :%s" % (str(len(exemplar))))
    self.exemplar_set.append(exemplar)
    # self.exemplar_set.append(images)
```