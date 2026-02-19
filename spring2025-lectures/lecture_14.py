from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np
import kenlm
import fasttext
import itertools
import mmh3
from bitarray import bitarray
from basic_util import count, repeat
from file_util import download_file
from execute_util import text, image, link
from lecture_util import article_link, named_link
from references import dolma

def main():
    text("Last lecture: overview of datasets used for training language models")
    text("- Live service (GitHub) → dump/crawl (GH Archive) → processed data (The Stack)")
    text("- Processing: HTML to text, language/quality/toxicity filtering, deduplication")

    text("This lecture: deep dive into the mechanics")
    text("- Algorithms for filtering (e.g., classifiers)")
    text("- Applications of filtering (e.g., language, quality, toxicity)")
    text("- Deduplication (e.g., Bloom filters, MinHash, LSH)")

    filtering_algorithms()
    filtering_applications()
    deduplication()

    text("### Summary")
    text("- Algorithmic tools: n-gram models (KenLM), classifiers (fastText), importance resampling (DSIR)")
    text("- Applications: language identification, quality filtering, toxicity filtering")
    text("- Deduplication: hashing scales to large datasets for fuzzy matching")
    text("- Now you have the tools (mechanics), just have to spend time with data (intuitions)")


def filtering_algorithms():
    text("Algorithmic building block:")
    text("- Given some **target data** T and lots of **raw data** R, find subset T' of R similar to T.") # 数据过滤的基本任务：给定一小组目标数据 T（比如高质量的论文、维基百科）和海量的原始数据 R（比如整个互联网抓取的网页），目标是从 R 中找出一个与 T 相似的子集 T'。
    image("images/raw-target-schema.png", width=600)

    text("Desiderata for filtering algorithm:")
    text("- Generalize from the target data (want T and T' to be different)") # 泛化能力：算法应该能从目标数据 T 中学习到某种“风格”或“分布特征”，并应用到未见过的大数据 R 上。理想情况下，T' 应该与 T 有统计上的相似性，但又不能完全照搬 T 的内容（否则就失去了过滤的意义）。
    text("- Extremely fast (have to run it on R, which is huge)") # 极高的效率：R 通常非常庞大（TB 甚至 PB 级别），算法必须在合理的时间内处理完 R 中的每一个数据项。

    kenlm_main()         # Train n-gram model
    fasttext_main()      # Train a classifier
    dsir_main()          # Train bag of n-grams model, do importance resampling
    filtering_summary()

    text("Survey paper on data selection "), link("https://arxiv.org/abs/2402.16827")


def kenlm_main():
    text("**n-gram model with Kneser-Ney smoothing** "), article_link("https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing") # 引入 Kneser-Ney 平滑算法。这是传统 n-gram 模型中最强大的平滑技术，用于处理训练中没见过的词组。
    text("- KenLM: fast implementation originally for machine translation "), named_link("code", "https://kheafield.com/code/kenlm/") # 介绍 KenLM。这是一个专为速度和内存效率设计的 C++ 库（提供 Python 接口），最初用于机器翻译。
    text("- Common language model used for data filtering") # 明确定位。KenLM 是大模型数据清洗中极常用的工具，因为它非常简单且极快（核心逻辑只是计数和归一化）。
    text("- Extremely simple / fast - just count and normalize")

    text("### Concepts")
    text("Maximum likelihood estimation of n-gram language model:")
    text("- n = 3: p(in | the cat) = count(the cat in) / count(the cat)") # 解释 n-gram 的最大似然估计。例如在 3-gram 中，已知前面两个词是 "the cat"，下一个词是 "in" 的概率，就是 "the cat in" 出现的次数除以 "the cat" 出现的次数。
    text("Problem: sparse counts (count of many n-grams is 0 for large n)") 
    text("Solution: Use Kneser-Ney smoothing to handle unseen n-grams "), article_link("https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing")
    text("- p(in | the cat) depends on p(in | cat) too") # 指出数据稀疏问题。如果某个短语在训练集里没出现过，概率就是 0，这显然不合理。Kneser-Ney 平滑通过结合更短的词组来“回退”并估算概率。

    # Download a KenLM language model
    model_url = "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/en.arpa.bin"
    model_path = "var/en.arpa.bin"
    download_file(model_url, model_path)
    model = kenlm.Model(model_path)

    # Use the language model
    def compute(content: str):
        # Hacky preprocessing。简单的预处理。给文本首尾加上句子开始 <s> 和结束 </s> 标记，并给标点符号加空格。
        content = "<s> " + content.replace(",", " ,").replace(".", " .") + " </s>"

        # log p(content)。model.score 计算该句子的对数概率（Log Probability）。分值越接近 0（即负数越小），代表句子在语言模型下越“正常”。
        score = model.score(content)

        # Perplexity normalizes by number of tokens to avoid favoring short documents
        num_tokens = len(list(model.full_scores(content))) # score 是累加的对数概率，为了公平比较长短句，需要除以词数 num_tokens。
        perplexity = math.exp(-score / num_tokens) # 困惑度是 $\exp(- \text{平均对数概率})$。困惑度越低，代表模型对这段文本的预测越有信心（文本越符合该语言的模式）。

        return score, perplexity

    score, perplexity = compute("Stanford University was founded in 1885 by Leland and Jane Stanford as a tribute to the memory of their only child, Leland Stanford Jr.")  # @inspect score, @inspect perplexity
    score, perplexity = compute("If you believe that the course staff made an objective error in grading, you may submit a regrade request on Gradescope within 3 days after the grades are released.")  # @inspect score, @inspect perplexity
    score, perplexity = compute("asdf asdf asdf asdf asdf")  # @inspect score, @inspect perplexity
    score, perplexity = compute("the the the the the the the the the the the the the the the the")  # @inspect score, @inspect perplexity

    text("### CCNet")
    link("https://arxiv.org/pdf/1911.00359")
    text("- Items are paragraphs of text")
    text("- Sort paragraphs by increasing perplexity")
    text("- Keep the top 1/3")
    text("- Was used in LLaMA")

    text("Summary: Kneser-Ney n-gram language models (with KenLM implementation) is fast but crude")


def fasttext_main():
    text("fastText classifier "), link("https://arxiv.org/pdf/1607.01759")
    text("- Task: text classification (e.g., sentiment classification)")
    text("- Goal was to train a fast classifier for text classification")
    text("- They found it was as good as much slower neural network classifiers")

    text("### Baseline: bag of words (not what they did)")
    L = 32                              # Length of input
    V = 8192                            # Vocabulary size
    K = 64                              # Number of classes
    W = nn.Embedding(V, K)              # Embedding parameters (V x K)
    x = torch.randint(V, (L,))          # Input tokens (L) - e.g., ["the", "cat", "in", "the", "hat"]
    y = softmax(W(x).mean(dim=0))       # Output probabilities (K)。将所有词的向量取平均，然后预测类别。
    text("Problem: V*K parameters (could be huge)")

    text("### fastText classifier: bag of word embeddings")
    H = 16                              # Hidden dimension
    W = nn.Embedding(V, H)              # Embedding parameters (V x H)
    U = nn.Linear(H, K)                 # Head parameters (H x K)
    y = softmax(U(W(x).mean(dim=0)))    # Output probabilities (K)。流程：取均值 -> 线性投影 -> 预测
    text("Only H*(V + K) parameters")

    text("Implementation:")
    text("- Parallelized, asynchronous SGD") # 异步随机梯度下降。
    text("- Learning rate: linear interpolation from [some number] to 0 "), article_link("https://github.com/facebookresearch/fastText/blob/main/src/fasttext.cc#L653") # 学习率随训练进度线性衰减到 0。

    text("### Bag of n-grams")
    x = ["the cat", "cat in", "in the", "the hat"]  # @inspect x
    text("Problem: number of bigrams can get large (and also be unbounded)") # 如果把所有的二元词组（bigrams）都放进词表，词表会爆炸。
    text("Solution: hashing trick") # 不再维护显式的词组词表，而是直接计算词组的哈希值并取模。即使有冲突，在极高维度下对精度的影响也很小，但却极大地节省了空间。
    num_bins = 8  # In practice, 10M bins
    hashed_x = [mmh3.hash(bigram) % num_bins for bigram in x]  # @inspect hashed_x

    text("- For quality filtering, we have K = 2 classes (good versus bad)")
    text("- In that case, fastText is just a linear classifier (H = K = 2)")

    text("In general, can use any classifier (e.g., BERT, Llama), it's just slower")


def dsir_main():
    text("Data Selection for Language Models via Importance Resampling (DSIR) "), link("https://arxiv.org/abs/2302.03169")
    image("https://www.jinghong-chen.net/content/images/size/w1200/2023/12/Screenshot-2023-12-24-at-17.41.38.png", width=600)

    importance_sampling()

    text("Setup:")
    text("- Target dataset D_p (small)")
    text("- Proposal (raw) dataset D_q (large)")

    text("Take 1:")
    text("- Fit target distribution p to D_p")
    text("- Fit proposal distribution q to D_q")
    text("- Do importance resampling with p, q, and raw samples D_q")
    text("Problem: target data D_p is too small to estimate a good model")

    text("Take 2: use hashed n-grams")
    training_text = "the cat in the hat"

    # Hash the n-grams
    num_bins = 4
    def get_hashed_ngrams(text: str): # 将一段文本拆开，计算每个词的哈希值并取模。
        ngrams = text.split(" ")  # Unigram for now
        return [mmh3.hash(ngram) % num_bins for ngram in ngrams]

    training_hashed_ngrams = get_hashed_ngrams(training_text)  # @inspect training_hashed_ngrams。把 "the cat in the hat" 变成了 [3, 1, 1, 3, 3]。

    # Learn unigram model
    probs = [count(training_hashed_ngrams, x) / len(training_hashed_ngrams) for x in range(num_bins)]  # @inspect probs。统计每个哈希桶在数据中出现的比例。这实际上是在训练一个基于哈希特性的简化语言模型。得到了目标分布 P（在哈希空间中）。这里是[0, 0.4, 0, 0.6]。

    # Evaluate probability of any sentence
    hashed_ngrams = get_hashed_ngrams("the text")  # @inspect hashed_ngrams。[3, 0]
    prob = np.prod([probs[x] for x in hashed_ngrams])  # @inspect prob。计算 "the text" 的概率。这里是 0.6 * 0 = 0。
    text("Result: DSIR slightly better than heuristic classification (fastText) on the [GLUE](https://gluebenchmark.com/) benchmark")
    image("images/dsir-results.png", width=700)
    
    text("Comparison with fastText:")
    text("- Modeling distributions is a more principled approach capturing diversity")
    text("- Similar computation complexity")
    text("- Both can be improved by better modeling")


def importance_sampling():
    text("Setup:")
    text("- Target distribution p (want samples from here)")
    text("- Proposal distribution q (have samples from here)")

    vocabulary = [0, 1, 2, 3]
    p = [0.1, 0.2, 0.3, 0.4]
    q = [0.4, 0.3, 0.2, 0.1]

    # 1. Sample from q
    n = 100
    samples = np.random.choice(vocabulary, p=q, size = n)  # @inspect samples
    text(f"Samples (q): {samples}")

    # 2. Compute weights over samples (w \propto p/q)
    w = [p[x] / q[x] for x in samples]  # @inspect w。对于取出的每一个样本 x，计算它的权重。如果 p(x) 很大（目标想要），而 q(x) 很小（原始数据里稀缺），那么 w 就会很大。
    z = sum(w)  # @inspect z
    w = [w_i / z for w_i in w]  # @inspect w

    # 3. Resample
    samples = np.random.choice(samples, p=w, size=n)  # @inspect samples。根据权重 w 重新采样。权重大的样本被选中的概率更大，从而把 q 的分布“拉向” p 的分布。
    text(f"Resampled (p): {samples}")


def filtering_summary():
    text("Implementations: KenLM, fastText, DSIR")

    text("### General framework")
    text("Given target T and raw R, find subset of R similar to T")
    text("1. Estimate some model based on R and T and derive a scoring function")
    text("2. Keep examples in R based on their score")

    text("### Instantiations of the framework")

    text("Generative model of T (KenLM):")
    text("1. score(x) = p_T(x)")
    text("2. Keep examples x with score(x) >= threshold (stochastically)")

    text("Discriminative classifier (fastText):")
    text("1. score(x) = p(T | x)")
    text("2. Keep examples x with score(x) >= threshold (stochastically)")

    text("Importance resampling (DSIR):")
    text("1. score(x) = p_T(x) / p_R(x)")
    text("2. Resample examples x with probability proportional to score(x)")


def filtering_applications():
    text("The same data filtering machinery can be used for different filtering tasks.")
    language_identification()
    quality_filtering()
    toxicity_filtering()


def language_identification():
    text("Language identification: find text of a specific language (e.g., English)")

    text("Why not just go multilingual?")
    text("- Data: difficult to do curation / processing of high-quality data in any given language")
    text("- Compute: in computed-limited regime, less compute/tokens dedicated to any given language")
    text("Models differ on multilinguality:")
    text("- English was only 30% of BLOOM (was undertrained), English performance suffered "), link("https://arxiv.org/pdf/2303.03915")
    text("- Most frontier models (GPT-4, Claude, Gemini, Llama, Qwen) are heavily multilingual (sufficiently trained)")

    text("fastText language identification "), article_link("https://fasttext.cc/docs/en/language-identification.html")
    text("- Off-the-shelf classifier")
    text("- Supports 176 languages")
    text("- Trained on multilingual sites: Wikipedia, Tatoeba (translation site) and SETimes (Southeast European news)")

    text("Example: Dolma keeps pages with p(English) >= 0.5 "), link(dolma)
    
    # Download the model
    model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    model_path = "var/lid.176.bin"
    download_file(model_url, model_path)
    model = fasttext.load_model(model_path)

    # Make predictions
    predictions = model.predict(["The quick brown fox jumps over the lazy dog."])  # English @inspect predictions
    predictions = model.predict(["The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."])  # Duplicate @inspect predictions
    predictions = model.predict(["OMG that movie was 🔥🔥! So dope 😎🤘!"])  # Informal English @inspect predictions
    predictions = model.predict(["Auf dem Wasser zu singen"])  # German @inspect predictions
    predictions = model.predict(["The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$."])  # Latex @inspect predictions
    predictions = model.predict(["for (int i = 0; i < 10; i++)"])  # C++ @inspect predictions
    predictions = model.predict(["Hello!"])  # English @inspect predictions
    predictions = model.predict(["Bonjour!"])  # French @inspect predictions
    predictions = model.predict(["Feliz Navidad / Próspero año y felicidad / I wanna wish you a Merry Christmas"])  # Spanish + English @inspect predictions

    text("Caveats:")
    text("- Difficult for short sequences") # 文本太短信息量不足
    text("- Difficult for low-resource languages") # 某些小语种数据少，识别不准
    text("- Could accidentally filter out dialects of English") # 可能会误伤英语方言
    text("- Hard for similar languages (Malay and Indonesian)") # 相似语言难以区分
    text("- Ill-defined for code-switching (e.g., Spanish + English)") # 混合语言定义模糊

    text("OpenMathText "), link("https://arxiv.org/pdf/2310.06786")
    text("- Goal: curate large corpus of mathematical text from CommonCrawl") # 目标：从 CommonCrawl 中筛选出数学文本
    text("- Use rules to filter (e.g., contains latex commands)") # 使用规则过滤（例如包含 latex 命令）
    text("- KenLM trained on ProofPile, keep if perplexity < 15000") # KenLM 在 ProofPile 上训练，保留困惑度 < 15000 的文本
    text("- Trained fastText classifier to predict mathematical writing, threshold is 0.17 if math, 0.8 if no math") # 训练 fastText 分类器来预测数学写作，阈值是 0.17 如果是数学，0.8 如果不是数学
    text("Result: produced 14.7B tokens, used to train 1.4B models that do better than models trained on 20x data") # 结果：生成了 14.7B tokens，用于训练 1.4B 模型，比训练 20x 数据的模型效果更好


def quality_filtering():
    text("- Some deliberately do not use model-based filtering (C4, Gopher, RefinedWeb, FineWeb, Dolma)")
    text("- Some use model-based filtering (GPT-3, LLaMA, DCLM) [becoming the norm]")

    text("**GPT-3** "), link("https://arxiv.org/pdf/2005.14165")  # Appendix A
    text("- Positives: samples from {Wikipedia, WebText2, Books1, Books2}")
    text("- Negatives: samples from CommonCrawl")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Probability_density_function_of_Pareto_distribution.svg/325px-Probability_density_function_of_Pareto_distribution.svg.png", width=0.5)
    text("Train linear classifier based on word features "), article_link("https://spark.apache.org/docs/latest/ml-features#tokenizer")
    text("Keep documents stochastically based on score")
    def keep_document(score: float) -> bool:
        return np.random.pareto(9) > 1 - score

    text("** LLaMA/RedPajama** "), link("https://arxiv.org/pdf/2302.13971")
    text("- Positives: samples from pages **referenced** by Wikipedia")
    text("- Negatives: samples from CommonCrawl")
    text("- Keep documents that are classified positive")

    text("**phi-1** "), link("https://arxiv.org/pdf/2306.11644")
    text("Philosophy: really high quality data (textbooks) to train a small model (1.5B)")
    text("Includes synthetic data from GPT 3.5 (later: GPT-4) and filtered data")

    R = "Python subset of the Stack"   # Raw data
    prompt = "determine its educational value for a student whose goal is to learn basic coding concepts"
    T = "Use GPT-4 with this prompt to classify 100K subset of R to get positive examples"
    text("Train random forest classifier on T using output embedding from pretrained codegen model")
    text("Select data from R that is classified positive by the classifier")

    text("Result on [HumanEval](https://huggingface.co/datasets/openai_humaneval):")
    text("- Train 1.3B LM on Python subset of The Stack (performance: 12.19% after 96K steps)")
    text("- Train 1.3B LM on new filtered subset (performance: 17.68% after 36K steps) - better!")


@dataclass
class Example:
    text: str
    label: int


def toxicity_filtering():
    # WARNING: potentially offensive content below
    text("Toxicity filtering in Dolma "), link(dolma)
    
    text("Dataset: Jigsaw Toxic Comments dataset (2018) "), named_link("dataset", "https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge")
    text("- Project goal: help people have better discussions online "), article_link("https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/46064")
    text("- Data: comments on Wikipedia talk page annotated with {toxic, severe_toxic, obscene, threat, insult, identity_hate}")

    text("Trained 2 fastText classifiers")
    text("- hate: positive = {unlabeled, obscene}, negative = all else")
    text("- NSFW: positive = {obscene}, negative = all else")

    # Examples from the dataset: (obscene, text)
    train_examples = [
        Example(label=0, text="Are you threatening me for disputing neutrality? I know in your country it's quite common to bully your way through a discussion and push outcomes you want. But this is not Russia."),
        Example(label=1, text="Stupid peace of shit stop deleting my stuff asshole go die and fall in a hole go to hell!"),
    ]

    # Download model
    model_url = "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin"
    model_path = "var/jigsaw_fasttext_bigrams_nsfw_final.bin"
    download_file(model_url, model_path)
    model = fasttext.load_model(model_path)

    # Make predictions
    predictions = model.predict([train_examples[0].text])  # @inspect predictions
    predictions = model.predict([train_examples[1].text])  # @inspect predictions
    predictions = model.predict(["I love strawberries"])  # @inspect predictions
    predictions = model.predict(["I hate strawberries"])  # @inspect predictions


def print_predict(model, content):
    """Run classifier `model` on `content` and print out the results."""
    predictions = model.predict([content])
    print(predictions)
    #labels, prob =
    #labels = ", ".join(labels)
    #text(f"{content} => {labels} {prob}")


def deduplication(): # 去重
    text("Two types of duplicates:")
    text("- Exact duplicates (mirror sites, GitHub forks) "), named_link("Gutenberg mirrors", "https://www.gutenberg.org/MIRRORS.ALL") # 精确重复：完全一模一样，比如镜像网站或代码分叉
    text("- Near duplicates: same text differing by a few tokens") # 近似重复：内容核心一致，但有少量词语或格式不同

    text("Examples of near duplicates:")
    text("- Terms of service and licenses "), named_link("MIT license", "https://opensource.org/license/mit") # 服务条款和许可证
    text("- Formulaic writing (copy/pasted or generated from a template) "), image("https://d3i71xaburhd42.cloudfront.net/4566c0d22ebf3c31180066ab23b6c445aeec78d5/5-Table1-1.png", width=600) # 模板化写作（如电商产品简介、新闻模板）
    text("- Minor formatting differences in copy/pasting") # 复制粘贴时的微小格式差异

    text("Product description repeated 61,036 times in C4") # C4 中重复了 61,036 次的产品描述
    text("'“by combining fantastic ideas, interesting arrangements, and follow the current trends in the field of that make you more inspired and give artistic touches. We’d be honored if you can apply some or all of these design in your wedding.  believe me, brilliant ideas would be perfect if it can be applied in real and make the people around you amazed!")
    named_link("example page", "https://www.amazon.co.uk/suryagede-100-Graffiti-Gas-Mask/dp/B07CRHT3RG")

    text("Deduplication training data makes language models better "), link("https://arxiv.org/pdf/2107.06499")
    text("- Train more efficiently (because have fewer tokens)") # 效率高：减少总数据量，训练更快
    text("- Avoid memorization (can mitigate copyright, privacy concerns)") # 避免记忆（可缓解版权、隐私问题）

    text("Design space:")
    text("1. What is an item (sentence, paragraph, document)?") # 处理单元：是以句子、段落还是全文为单位？
    text("2. How to match (exact match, existence of common subitem, fraction of common subitems)?") # 匹配方式：精确匹配还是看子序列的重合比例？
    text("3. What action to take (remove all, remove all but one)?") # 处理动作：删掉所有重复项，还是保留一份？

    text("Key challenge:")
    text("- Deduplication is fundamentally about comparing items to other items")
    text("- Need linear time algorithms to scale")

    hash_functions()

    exact_deduplication()
    bloom_filter()

    jaccard_minhash()
    locality_sensitive_hashing()


def hash_functions():
    text("- Hash function h maps item to a hash value (integer or string)")
    text("- Hash value much smaller than item")
    text("- Hash collision: h(x) = h(y) for x ≠ y")

    text("Tradeoff between efficiency and collision resistance "),  article_link("https://softwareengineering.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed")
    text("- Cryptographic hash functions (SHA-256): collision resistant, slow (used in bitcoin)")
    text("- DJB2, MurmurHash, CityHash: not collision resistant, fast (used for hash tables)")

    text("We will use MurmurHash:")
    h = mmh3.hash("hello")  # @inspect h


def exact_deduplication():
    text("**Simple example**")
    text("1. Item: string")
    text("2. How to match: exact match")
    text("3. Action: remove all but one")

    # Original items
    items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]  # @inspect items

    # Compute hash -> list of items with that hash
    hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)

    # Keep one item from each group
    deduped_items = [next(group) for h, group in hash_items]  # @inspect deduped_items

    text("- Pro: simple, clear semantics, high precision")
    text("- Con: does not deduplicate near duplicates")
    text("- This code is written in a MapReduce way, can easily parallelize and scale")

    text("**C4** "), link("https://arxiv.org/pdf/1910.10683v4")
    text("1. Item: 3-sentence spans")
    text("2. How to match: use exact match")
    text("3. Action: remove all but one")
    text("Warning: when a 3-sentence span is removed from the middle of a document, the resulting document might not be coherent")


def bloom_filter():
    text("Goal: efficient, approximate data structure for testing set membership")

    text("Features of Bloom filters")
    text("- Memory efficient")
    text("- Can update, but can't delete")
    text("- If return 'no', definitely 'no'")
    text("- If return 'yes', most likely 'yes', but small probability of 'no'")
    text("- Can drive the false positive rate down exponentially with more time/compute")

    items = ["the", "cat", "in", "the", "hat"]
    non_items = ["what", "who", "why", "when", "where", "which", "how"]

    text("First, make the range of hash function small (small number of bins).")
    m = 8  # Number of bins
    table = build_table(items, m)
    for item in items:
        assert query_table(table, item, m) == 1
    result = {item: query_table(table, item, m) for item in non_items}  # @inspect result
    num_mistakes = count(result.values(), True)  # @inspect num_mistakes
    false_positive_rate = num_mistakes / (len(items) + num_mistakes)  # @inspect false_positive_rate
    text("Problem: false positives for small bins")

    text("Naive solution: increase the number of bins")
    text("Error probability is O(1/num_bins), decreases polynomially with memory")

    text("Better solution: use more hash functions")
    k = 2  # Number of hash functions
    table = build_table_k(items, m, k)
    for item in items:
        assert query_table_k(table, item, m, k) == 1
    result = {item: query_table_k(table, item, m, k) for item in non_items}  # @inspect result
    num_mistakes = count(result.values(), 1)  # @inspect num_mistakes
    false_positive_rate = num_mistakes / (len(items) + num_mistakes)  # @inspect false_positive_rate
    text("Reduced the false positive rate!")

    false_positive_rate_analysis()


def false_positive_rate_analysis():
    text("Assume independence of hash functions and items "), article_link("https://en.wikipedia.org/wiki/Bloom_filter")
    m = 1000   # Number of bins
    k = 10     # Number of hash functions
    n = 100    # Number of items we're inserting

    text("Consider a test input (not in the set) that would hash into a given test bin (say, i).")
    text("Now consider putting items into the Bloom filter and seeing if it hits i.")

    # Insert one item, ask if the test bin B(i) = 1?
    # B: [0 0 1 0 0 0 0 0 0 0] - have to miss 1 time
    f = 1 / m                              # P[B(i) = 1 after 1 insertion with 1 hash function]  # @inspect f # 插入 1 个项，只用 1 个哈希函数时，位置 i 被命中的概率
    # B: [0 0 1 0 0 1 0 1 0 0] - have to miss k times
    f = 1 - (1 - 1 / m) ** k               # P[B(i) = 1 after 1 insertion with k hash functions]  # @inspect f # 插入 1 个项，用 k 个哈希函数后，位置 i 至少被命中一次的概率

    # Insert n items, ask if the test bin B(i) = 1?
    # Have to miss k*n times
    f = 1 - (1 - 1 / m) ** (k * n)         # P[B(i) = 1 after n insertions for 1 hash function]  # @inspect f # 插入 n 个项（共 k*n 次哈希）后，位置 i 为 1 的概率
    # Get k chances to miss (since test input is hashed k times too)
    f = f ** k                             # P[B(i) = 1 after n insertions for k hash functions]  # @inspect f # 一个新项（不在集内）被误判为“已存在”的概率

    text("Optimal value of k (given fixed m / n ratio) [results in f ~ 0.5]")
    k = math.log(2) * m / n  # @inspect k # 在给定内存 m 和数据量 n 的前提下，能使误报率 f 最小的最优哈希个数 k
    text("Resulting false positive rate (improved)")
    f = 0.5 ** k  # @inspect f # 使用最优 k 时，误报率的近似计算

    text("Tradeoff between compute (k), memory (m), and false positive rate (f) "), named_link("lecture notes", "https://people.eecs.berkeley.edu/~daw/teaching/cs170-s03/Notes/lecture10.pdf")

    text("Example: Dolma")
    text("- Set false positive rate to 1e-15")
    text("- Perform on items = paragraphs")


def build_table(items: list[str], num_bins: int):
    """Build a Bloom filter table of size `num_bins`, inserting `items` into it."""
    table = bitarray(num_bins)  # @inspect table
    for item in items:
        h = mmh3.hash(item) % num_bins  # @inspect item, @inspect h
        table[h] = 1  # @inspect table
    return table


def build_table_k(items: list[str], num_bins: int, k: int):
    """Build a Bloom filter table of size `num_bins`, inserting `items` into it.
    Use `k` hash functions."""
    table = bitarray(num_bins)  # @inspect table
    for item in items:
        # For each of the k functions
        for seed in range(k):
            h = mmh3.hash(item, seed) % num_bins  # @inspect item, @inspect h, @inspect seed
            table[h] = 1  # @inspect table
    return table


def query_table(table: bitarray, item: str, num_bins: int, seed: int = 0):
    """Return whether `item` is in the `table`."""
    h = mmh3.hash(item, seed) % num_bins
    return table[h]


def query_table_k(table: bitarray, item: str, num_bins: int, k: int):
    """Return 1 if table set to 1 for all `k` hash functions."""
    return int(all(
        query_table(table, item, num_bins, seed)
        for seed in range(k)
    ))


def jaccard_minhash():
    text("Let's now look at approximate set membership.")
    text("First we need a similarity measure.")

    text("### Jaccard similarity")
    text("Definition: Jaccard(A, B) = |A intersect B| / |A union B|")
    A = {"1", "2", "3", "4"}
    B = {"1", "2", "3", "5"}

    def compute_jaccard(A, B):
        intersection = len(A & B)  # @inspect intersection
        union = len(A | B)  # @inspect union
        return intersection / union
    jaccard = compute_jaccard(A, B)  # @inspect jaccard

    text("Definition: two documents are **near duplicates** if their Jaccard similarity >= threshold")

    text("Algorithmic challenge: find near duplicates in linear time")

    text("### MinHash")
    text("MinHash: a random hash function h so that Pr[h(A) = h(B)] = Jaccard(A, B)")

    text("Normally, you want different items to hash to different hashes")
    text("...but here, you want collision probability to depend on similarity")

    def minhash(S: set[str], seed: int):
        return min(mmh3.hash(x, seed) for x in S)

    text("Characteristic matrix representation:")
    text("item | A | B", verbatim=True)
    text("1    | 1 | 1", verbatim=True)
    text("2    | 1 | 1", verbatim=True)
    text("3    | 1 | 1", verbatim=True)
    text("4    | 1 | 0", verbatim=True)
    text("5    | 0 | 1", verbatim=True)

    text("Random hash function induces a permutation over items")
    text("Look at which item is first in A and which item is first in B.")
    text("Each item has the same probability as being first (min)")
    text("- If 1, 2, 3 is first, then first in A = first in B.")
    text("- If 4, 5 is first, then first in A ≠ first in B.")

    # Verify MinHash approximates Jaccard as advertised
    n = 100  # Generate this many random hash functions
    matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]
    estimated_jaccard = count(matches, True) / len(matches)  # @inspect estimated_jaccard
    assert abs(estimated_jaccard - jaccard) < 0.01

    text("Now we can hash our items, but a collision doesn't tell us Jaccard(A, B) > threshold.")


def locality_sensitive_hashing():
    text("Locality sensitive hashing (LSH) "), named_link("book chapter", "http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf")

    text("Suppose we hash examples just one MinHash function")
    text("P[A and B collide] = Jaccard(A, B)")
    text("On average, more similar items will collide, but very stochastic...")

    text("Goal: have A and B collide if Jaccard(A, B) > threshold")
    text("We have to somehow sharpen the probabilities...")

    text("Solution: use n hash functions")
    text("Break up into b bands of r hash functions each (n = b * r)")

    n = 12      # Number of hash functions
    b = 3       # Number of bands
    r = 4       # Number of hash functions per band
    text("Hash functions:")
    text("h1 h2 h3 h4  |  h5 h6 h7 h8  |  h9 h10 h11 h12", verbatim=True)

    text("Key: A and B collide if for *some* band, *all* its hash functions return same value")
    text("As we will see, the and-or structure of the bands sharpens the threshold")

    text("Given Jaccard(A, B), what is the probability that A and B collide?")

    def get_prob_collision(sim, b, r):  # @inspect sim, @inspect b, @inspect r
        prob_match = sim ** r                        # Probability that a fixed band matches  @inspect prob_match
        prob_collision = 1 - (1 - prob_match) ** b   # Probability that some band matches  @inspect prob_collision
        return prob_collision

    text("**Example**")
    prob_collision = get_prob_collision(sim=0.8, b=5, r=10)  # @inspect prob_collision
    image("https://cdn.sanity.io/images/vr8gru94/production/b470799575b8e77911bacb8500977afef06d6c85-1280x720.png", width=600)


    sims = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
    probs = {sim: get_prob_collision(sim=sim, b=10, r=10) for sim in sims}  # @inspect probs

    text("Increasing r sharpens the threshold and moves the curve to the right (harder to match)")
    probs = {sim: get_prob_collision(sim=sim, b=10, r=20) for sim in sims}  # @inspect probs

    text("Increasing b moves the curve to the left (easier to match)")
    probs = {sim: get_prob_collision(sim=sim, b=20, r=20) for sim in sims}  # @inspect probs
    image("https://cdn.sanity.io/images/vr8gru94/production/aace49fa240778e8ecf6e85ad08a2de7f5385566-1280x720.png", width=600)

    text("Example setting "), link("https://arxiv.org/pdf/2107.06499"), text(": n = 9000, b = 20, r = 450")
    b = 20
    r = 450
    text("What is the threshold (where the phase transition happens)?")
    threshold = (1 / b) ** (1 / r)  # @inspect threshold
    text("Probability that a fixed band matches:")
    prob_match = (1 / b)  # @inspect prob_match
    text("Probability that A and B collide (≈ 1-1/e):")
    prob_collision = 1 - (1 - 1 / b) ** b  #  @inspect prob_collision


if __name__ == "__main__":
    main()
