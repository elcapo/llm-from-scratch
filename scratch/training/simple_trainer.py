from typing import Optional
import torch
from ..gpt_model import GptModel
from ..tokenizers.base_tokenizer import BaseTokenizer
from ..generators.base_generator import BaseGenerator
from ..generators.simple_text_generator import SimpleTextGenerator
from .batch_loss import BatchLoss
from .loader_loss import LoaderLoss

class SimpleTrainer:
    def __init__(
        self,
        model: GptModel,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        tokenizer: BaseTokenizer,
        device: Optional[str] = None,
        generator: Optional[BaseGenerator] = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.batch_loss = BatchLoss(model, device)
        self.loader_loss = LoaderLoss(model, device)
        if generator is None:
            generator = SimpleTextGenerator(model, max_new_tokens=50)
        self.generator = generator

    def train(self, num_epochs: int, eval_freq: int, eval_iter: int, start_context: str):
        train_losses, validation_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        for epoch in range(num_epochs):
            self.model.train()

            if global_step % eval_freq == 0:
                print()
                print("Epoch", epoch + 1)

            for input_batch, target_batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.batch_loss.calc(input_batch, target_batch)
                loss.backward()
                self.optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, validation_loss = self.evaluate_model(eval_iter)
                    train_losses.append(train_loss)
                    validation_losses.append(validation_loss)
                    track_tokens_seen.append(tokens_seen)

                    print()
                    print("- Step", global_step)
                    print("- Train loss", train_loss)
                    print("- Evaluation loss", validation_loss)
            
            self.generate_and_print_sample(start_context)

        print()
        print("Training finished")

        return train_losses, validation_losses, track_tokens_seen

    def evaluate_model(self, eval_iter: int):
        self.model.eval()
        with torch.no_grad():
            train_loss = self.loader_loss.calc(self.train_loader, num_batches=eval_iter)
            validation_loss = self.loader_loss.calc(self.validation_loader, num_batches=eval_iter)
        self.model.train()
        return train_loss, validation_loss

    def generate_and_print_sample(self, start_context: str):
        self.model.eval()
        context_size = self.model.positional_embeddings.weight.shape[0]
        encoded = self.tokenizer.text_to_tokens(start_context).to(self.device)
        with torch.no_grad():
            tokens = self.generator.generate(encoded)
        decoded_text = self.tokenizer.tokens_to_text(tokens)
        print("- Generated: " + decoded_text.replace("\n", " "))
        self.model.train()
