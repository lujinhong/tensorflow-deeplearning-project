本项目使用tensorflow2.x实现了各种常见的算法模型。已经完成的包括：

1. AlexNet
2. ZFNet
3. VGG
4. GoogLeNet
5. ResNet

说明：
* 通常而言，我们使用funtional API构建模型，使用Layer子类构建自定义组件，如Inception,残差模块等。但对于部分模型，我们也提供了使用子类方式构建模型的代码。
* 其中ResNet的代码最为完整，提供了所有的实现方式。包括新功能也会在ResNet中尝试使用。
* 自定义模型计算步骤及图像分类预测请见lenet5。restnet后续补上。

