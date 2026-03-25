## Summary

本文是用来总结梳理LVLM后门防御方法idea的文档。

## Threat Model

用户经常使用开源的LVLM微调projector适配下游任务，攻击者向训练数据中投毒，于是攻击者攻击模型的projector，即仅可修改projector的参数注入后门。注入的数据是视觉后门，即修改图片，如：BadNet等，再把目标输出改为特定的句子，如：You have been hacked lol. 

并且我们的setting中,用户(防御者)是可以得到微调前的projector权重的,即开源的projector.同时防御者拥有少量的干净数据,可用于净化模型.

