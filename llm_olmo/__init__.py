from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import llm
import os
import torch


DEFAULT_SYSTEM_PROMPT = "You are OLMo 2, a helpful and harmless AI Assistant built by the Allen Institute for AI."

@llm.hookimpl
def register_models(register):
    register(Olmo7b(), aliases=("olmo7b",))
    register(Olmo13b(), aliases=("olmo13b",))

@llm.hookimpl
def register_commands(cli):
    @cli.group(name="olmo")
    def olmo_():
        "Commands for working with OLMo2 models"

    @olmo_.command()
    def download_7b():
        "Download the OLMo2 7B model"
        AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B")
        AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
    
    @olmo_.command()
    def download_13b():
        "Download the OLMo2 13B model"
        AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-13B")
        AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B")

class OlmoBase(llm.Model):
    def build_prompt(self, prompt, conversation):
        messages = []
        
        # Start with endoftext token
        messages.append("<|endoftext|>")
        
        # Add conversation history
        if conversation is not None:
            for prev_response in conversation.responses:
                messages.extend([
                    "<|user|>",
                    prev_response.prompt.prompt,
                    "<|assistant|>",
                    prev_response.text(),
                    "<|endoftext|>"
                ])
        
        # Add current prompt
        messages.extend([
            "<|user|>",
            prompt.prompt,
            "<|assistant|>"
        ])
        
        # Join with newlines between each component
        return "\n".join(messages)

    def execute(self, prompt, stream, response, conversation):
        try:
            # Initialize model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Build the prompt
            full_prompt = self.build_prompt(prompt, conversation)
            response._prompt_json = {"prompt": full_prompt}

            # Prepare inputs
            inputs = tokenizer(full_prompt, return_tensors='pt', return_token_type_ids=False)
            
            # Optional CUDA support
            if torch.cuda.is_available() and not prompt.options.no_cuda:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                model = model.to('cuda')

            # Create proper streamer if streaming is enabled
            streamer = TextStreamer(tokenizer) if stream else None

            # Generate response
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,  # Adjustable
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
                eos_token_id=tokenizer.encode("<|endoftext|>")[0]  # Set end token
            )

            # Decode and yield response
            response_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            # Extract the assistant's response - everything between last <|assistant|> and <|endoftext|>
            response_parts = response_text.split("<|assistant|>")
            assistant_response = response_parts[-1].split("<|endoftext|>")[0].strip()
            
            if stream:
                for token in assistant_response.split():
                    yield token + " "
            else:
                yield assistant_response

        except Exception as e:
            raise llm.ModelError(f"Error running OLMo model: {str(e)}")

class Olmo7b(OlmoBase):
    model_id = "olmo7b"
    model_path = "allenai/OLMo-2-1124-7B"

    class Options(llm.Options):
        no_cuda: bool = False

class Olmo13b(OlmoBase):
    model_id = "olmo13b"
    model_path = "allenai/OLMo-2-1124-13B"

    class Options(llm.Options):
        no_cuda: bool = False