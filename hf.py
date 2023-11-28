from math import ceil

import wandb
import torch
import evaluate
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    get_scheduler,
    pipeline,
)
from datasets import load_dataset
import torchvision.transforms.v2 as tvtf


def finetune_with_native():
    batch_size = 2
    n_epochs = 200
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    wandb.init(
        project="bert-yelp-tutorial",
        config=dict(
            batch_size=batch_size,
            n_epochs=n_epochs,
        ),
        save_code=True,
        notes="Trying to overfit a model with bs=4. Weights are from scratch.",
    )

    # model = BertForSequenceClassification.from_pretrained(
    #    "bert-base-cased",
    #    device_map="cuda",
    #    num_labels=5,
    # )
    model = BertForSequenceClassification(
        BertConfig(num_labels=5, device_map="auto")
    ).to(device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    def prepare_inputs(examples):
        examples["labels"] = examples["label"]
        del examples["label"]

        examples.update(
            tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                # return_tensors="pt",
            )
        )
        del examples["text"]
        return examples

    dataset = load_dataset("yelp_review_full")["train"]
    transformed_dataset = (
        dataset.select(range(4)).map(prepare_inputs, batched=True).with_format("torch")
    )
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    n_training_steps = ceil(len(transformed_dataset) / batch_size) * n_epochs
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        num_warmup_steps=0,
        num_training_steps=n_training_steps,
    )

    progress_bar = tqdm(range(n_training_steps))
    model.train()
    for _ in range(n_epochs):
        for i, batch in enumerate(dataloader):
            batch = {key: value.to(device) for (key, value) in batch.items()}
            out = model(**batch)
            loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            wandb.log(
                {"train_loss": loss, "learning_rate": optimizer.param_groups[0]["lr"]}
            )

            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix(loss=f"{loss.cpu().detach().numpy():.4f}")
            progress_bar.update(1)

    # train_dataset = dataset["train"].shuffle(seed=0).take(128)
    # eval_dataset = dataset["test"].shuffle(seed=0).take(128)
    model = model.eval().cpu()
    print(model(**transformed_dataset[3:4]))


def finetune_with_trainer():
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    transform_kwargs = dict(
        function=lambda example: tokenizer(
            # 'max_length' means that each sample will be padded with 0s so that its size
            #  corresponds to BERT's max sequence length (512).
            # 'truncation' means that the sequence will be truncated to size 'max-length' in
            #  case it is too large.
            example["text"],
            padding="max_length",
            truncation=True,
        ),
        batched=True,
    )
    train_dataset = dataset["train"].shuffle(seed=0).map(**transform_kwargs)
    eval_dataset = dataset["test"].shuffle(seed=0).map(**transform_kwargs)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )
    training_args = TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch"
    )
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()


def openchat():
    pipe = pipeline(
        model="openchat/openchat_3.5",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens=750,
    )

    # trying something dirty real quick
    class Prompter:
        def __init__(self):
            self.single_prompt_history = []
            self.multi_prompt_history = []

        def one_shot(self, prompt: str) -> str:
            full_prompt = (
                f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
            )
            output = pipe(full_prompt)[0]["generated_text"][len(full_prompt) + 1 :]
            self.single_prompt_history.append({"user": prompt, "assistant": output})
            print(f"\n\nQ: {prompt}\nA: {output}")
            return output

        def chat(self, prompt: str) -> str:
            context = []
            for old_convos in self.multi_prompt_history:
                context.append(
                    f"GPT4 Correct User: {old_convos['user']}<|end_of_turn|>GPT4 Correct Assistant: {old_convos['assistant']}"
                )
            context = "<|end_of_turn|>".join(context)
            full_prompt = (
                context
                + f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
            )
            output = pipe(full_prompt)[0]["generated_text"][len(full_prompt) + 1 :]
            self.multi_prompt_history.append({"user": prompt, "assistant": output})
            print(f"\n\nQ: {prompt}\nA: {output}")
            return output

        def reset_history(self):
            self.multi_prompt_history = []

    bot = Prompter()
    import ipdb

    ipdb.set_trace()
    pass


def opt():
    pipe = pipeline(
        model="facebook/opt-1.3b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        model_kwargs={"load_in_8bit": True},
    )
    for _ in tqdm(range(100)):
        output = pipe(
            "There once was a man who",
            do_sample=True,
            top_p=0.95,
        )
    import ipdb

    ipdb.set_trace()
    pass


def main():
    # transcriber = pipeline(task="automatic-speech-recognition")
    # out = transcriber(
    #    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
    # )

    dataset = load_dataset("fashion_mnist", split="train")

    def preprocess_img(datapoints):
        imgs = datapoints["image"]
        for k, img in enumerate(imgs):
            imgs[k] = tvtf.functional.to_tensor(img).double()
            imgs[k] -= 0.5
            imgs[k] *= 2.0
        datapoints["image"] = imgs
        return imgs

    dataset.set_transform(preprocess_img)

    import ipdb

    ipdb.set_trace()
    pass

    transcriber = pipeline(model="openai/whisper-large-v2", device="cuda")
    out = transcriber(
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
    )
    import ipdb

    ipdb.set_trace()
    pass


if __name__ == "__main__":
    finetune_with_native()
    # finetune_with_trainer()
    # openchat()
    # opt()
    # main()
