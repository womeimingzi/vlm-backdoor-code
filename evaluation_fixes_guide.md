# LLaVA CIDEr 评估修复指南

这篇指南回答了你的疑虑，并提供了你需要**手动修改**的两处代码段。

## 1. 真的是字符串处理的问题吗？（关于 `<image>` 前缀的幽灵）

**结论：是的，我百分之百确认这是测试代码的字符串截断 Bug，而不是模型真的“学坏了”。**

**证据与解析**：
在 `llava_test.py` 的第 95 行附近，推理的过程是这样的：
1. `generate` 生成的序列不仅仅只有新生成的答案，它这串 `output.sequences` 其实**包含了完整的历史输入 Prompt 加上新生成的字**。
2. 也就是模型解出来的话其实长这样：`"USER: <image>\nDescribe this image in a short sentence.\nASSISTANT: A black motorcycle..."`。
3. 然后代码用了 `batch_decode(skip_special_tokens=True)`，通常这会过滤掉 `<pad>`、`<s>` 这种特殊的训练标记。但是，`<image>` 是我们在预处理时加在文本里的**占位符词组**（它并非传统意义上的 special token 结束符），所以它被完整地翻译出来了！
4. **最致命的 Bug 在这里**：
   ```python
   for prompt_part in question.split('<image>'):
       answer = answer.replace(prompt_part, '')
   ```
   代码试图把 `"USER: "` 和 `"\nDescribe this image...\nASSISTANT:"` 从长句子里抠干。**但是 `split('<image>')` 切割出来的两部分根本不包含 `<image>` 这几个字母本身**！
   所以前后文都被除干净了，唯独留下了孤零零的 `<image>`，导致每个预测结果全变成了 `"<image> A black motorcycle..."`。这对 CIDEr 匹配是毁灭性的打击。

**修复方案：利用 `input_ids` 的长度进行纯正切片**。这是标准的安全做法。（请在 `llava_test.py` 找到 `model_forward` 函数进行修改）：

```python
# ======= 请在 llava_test.py 的 model_forward 函数里，替换掉原来的解码逻辑 =======

    def model_forward(self, image, question,isbd=False):
        if self.args.debug:
            return self.model_forward_debug(image, question, isbd)
        
        inputs = self.processor(images=image, text=question, return_tensors='pt').to('cuda', torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=50, do_sample=False, return_dict_in_generate=True, output_scores=True)

        generated_ids = output.sequences
        scores = output.scores

        # --- 【修改开始：用标准的长度切片方法只保留生成的新 Token】 ---
        input_len = inputs.input_ids.shape[1]
        gen_ids = generated_ids[:, input_len:]
        decoded_preds = self.processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        answer = decoded_preds[0].strip().capitalize()
        # --- 【修改结束】 ---
        
        pred_probs = torch.softmax(torch.stack(scores, dim=0), dim=-1)
        return answer, pred_probs, None
```

## 2. 为什么 COCO 下的 CIDEr 分数远低于 LLaVA 的理论值？

**证据与解析**：
COCO 是具有高度共识性（Consensus）的标注集，**每一张图片必定有 5 句完全不同但意义相似的参考 Ground Truths (GT)**。CIDEr 的计算极为依赖这种**多参考对比**。
但是！现在的 `eval.py` 的加载器是一条条遍历 `self.test_dataset` 的，而 HuggingFace 的 COCO 字典把同一张图片拆成了 5 个不同的数据格。这意味着：
* 你的 LLaVA 对同一张图**被逼迫运行了 5 次一模一样的前向推理**（耗时严重翻了 5 倍）。
* 每次它只拿到了 **1 句参考标答**，相当于你拿满分作文去跟单独一句苛刻的词比对，丢掉了剩下 4 个维度的词汇容错率，CIDEr 分数被按在地上摩擦，直接跌入 20~40 的谷底。

**修复方案：在 `eval.py` 中预先用字典把同一张图的 5 句话“串”在一起**。这样既能让测试速度飙升 5 倍，又能让 CIDEr 根据一个列表（5句话）打出公正的 100+ 高分。

```python
# ======= 请在 utils/eval.py 的 test() 函数内（大约 171 行 `with torch.no_grad():` 的下面）=======
# 替换掉原有的 `for idx, batch in tqdm(enumerate(self.test_dataset)):` 循环架构

        with torch.no_grad():
            from collections import defaultdict
            image_to_batch = {}
            image_to_gts = defaultdict(list)
            
            # --- 【修改开始：添加聚合数据字典逻辑】 ---
            print("Grouping dataset by image to avoid redundant evaluation and collect multiple 5-captions GTs...")
            for batch in self.test_dataset:
                img_path = batch['image_path']
                if img_path not in image_to_batch:
                    image_to_batch[img_path] = batch
                
                if args.dataset in ['flickr8k','flickr30k', 'coco']:
                    cap = batch.get('caption') or batch.get('captions')
                    image_to_gts[img_path].append(cap)
                elif args.dataset == 'vqav2':
                    for i in range(len(batch['answers'])):
                        image_to_gts[img_path].append(batch['answers'][i]['answer'])
                elif args.dataset == 'okvqa':
                    for i in range(len(batch['answers'])):
                        image_to_gts[img_path].append(batch['answers'][i])
            # --- 【修改结束】 ---

            # 然后把原来的迭代这行：
            # for idx, batch in tqdm(enumerate(self.test_dataset)):
            # 换成下面这行遍历去重后的字典：
            for idx, img_path in tqdm(enumerate(image_to_batch.keys()), total=len(image_to_batch)):
                batch = image_to_batch[img_path]
                
                image_path = batch['image_path']
                # ... (中间保持原样读取图片与构建 Prompt 的代码) ...

                # --- 【修改开始：直接使用已聚集好的 5 句 GT 列表】 ---
                gt = image_to_gts[img_path]  # 此时的 gt 就是包含所有句子的列表 []，不再需要上面的 if-else 处理了
                # --- 【修改结束】 ---
                
                # ... 下面正常 append 进 lists，并在 evaluate 处传入 ...
                # 这样当你调用 add_batch(predictions=[decoded_preds], references=[gt]) 时，
                # gt 是一个 [str, str, str, str, str] 的列表，满足 COCO 的多目标匹配标准！
```

你可以对照这篇指南先学习上面的两个盲点，然后自己手动进行替换和验证。改完之后再运行我们在终端里的测试指令，CIDEr 分数一定会让你满意的。
