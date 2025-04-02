# Specifications for the implementation of DeepONet

## Structure and organization of the code

Use object-oriented programming with clean hierarchy of classes

-   base classes for model, data (feed into the branch net and the trunk net of DeepONet), data loader, trainer, etc
-   For the model, it implements the DeepONet with one branch net and one trunk net, which joint to produce the output.