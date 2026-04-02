from search import PaperSearcher

# openai/qwen/local
searcher = PaperSearcher('papers/iclr2026_papers.json', model_type='local')

# Or use OpenAI (better quality)
# searcher = PaperSearcher('iclr2026_papers.json', model_type='openai')

searcher.compute_embeddings()

examples = [
    {
        "title": "TTRL: Test-Time Reinforcement Learning",
        "abstract": "This paper investigates Reinforcement Learning (RL) on data without explicit labels for reasoning tasks in Large Language Models (LLMs). The core challenge of the problem is reward estimation during inference while not having access to ground-truth information. While this setting appears elusive, we find that common practices in Test-Time Scaling (TTS), such as majority voting, yield surprisingly effective rewards suitable for driving RL training. In this work, we introduce Test-Time Reinforcement Learning (TTRL), a novel method for training LLMs using RL on unlabeled data. TTRL enables self-evolution of LLMs by utilizing the priors in the pre-trained models. Our experiments demonstrate that TTRL consistently improves performance across a variety of tasks and models. Notably, TTRL boosts the pass@1 performance of Qwen-2.5-Math-7B by approxi- mately 211% on the AIME 2024 with only unlabeled test data. Furthermore, although TTRL is only supervised by the maj@n metric, TTRL has demonstrated performance to consistently surpass the upper limit of the initial model maj@n, and approach the performance of models trained directly on test data with ground-truth labels. Our experimental findings validate the general effectiveness of TTRL across various tasks and highlight TTRL’s potential for broader tasks and domains."
    },
]

results = searcher.search(examples=examples, top_k=100)

searcher.display(results, n=10)
searcher.save(results, 'results.json')

