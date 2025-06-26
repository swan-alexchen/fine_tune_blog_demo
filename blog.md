
# Fine-Tuning Llama3 Models on Nebula Block GPU Instances with Unsloth: A Complete Guide

Fine-tuning large language models like Llama3 has become essential for creating domain-specific AI applications. With Nebula Block's powerful GPU instances and Unsloth's optimization library, you can fine-tune Llama3 models **2x faster** and use **50% less memory** compared to traditional methods. This guide will walk you through the entire process.

## Why Unsloth + Nebula Block is the Perfect Combination

**Unsloth Benefits:**
- 2x faster training speeds
- 50% less memory usage
- Zero accuracy degradation
- Support for 4-bit and 16-bit quantization
- Seamless integration with Hugging Face

**Nebula Block Benefits:**
- On-demand NVIDIA H100, A100, and RTX GPUs
- Easy instance management via API or web UI
- S3-compatible storage for datasets and models
- Cost-effective pay-as-you-use pricing

## Prerequisites

Before starting, ensure you have:

1. **Nebula Block Account**: Sign up at [https://nebulablock.com/register](https://nebulablock.com/register) and receive $1 in free credits
2. **API Key**: Create one through the [customer portal](https://nebulablock.com/home) or via API
3. **SSH Key**: Upload your public SSH key for secure instance access
4. **Fine-tuning Dataset**: Prepared in chat/instruction format

## Step 1: Set Up Your Nebula Block Environment

### Create an API Key
```bash
curl -X POST '{API_URL}/api-keys' \
  -H 'Authorization: Bearer {TOKEN}' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "unsloth-llama3-key",
    "description": "API key for Llama3 fine-tuning with Unsloth"
  }'
```

### Upload SSH Key
```bash
curl -X POST '{API_URL}/ssh-keys' \
  -H 'Authorization: Bearer {TOKEN/KEY}' \
  -H 'Content-Type: application/json' \
  -d '{
    "key_name": "Unsloth Fine-tuning SSH Key",
    "key_data": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQD5..."
  }'
```

## Step 2: Choose the Right GPU Instance

With Unsloth's memory optimizations, you can use smaller GPU instances:

- **Llama3-8B**: RTX 4090 (24GB) or RTX 3090 (24GB) - **Now sufficient with Unsloth!**
- **Llama3-70B**: Single A100 (40GB) or H100 (80GB) - **Previously required multiple GPUs**
- **Llama3-405B**: H100 (80GB) with gradient checkpointing

Check available products:
```bash
curl -X GET '{API_URL}/computing/products' \
  -H 'Authorization: Bearer {TOKEN/KEY}'
```

## Step 3: Launch Your GPU Instance

```bash
curl -X POST '{API_URL}/computing/instance' \
  -H 'Authorization: Bearer {TOKEN/KEY}' \
  -H 'Content-Type: application/json' \
  -d '{
    "instance_name": "unsloth-llama3-training",
    "product_id": "fcp66d566c43a9501001256",
    "image_id": "fcp66d566c93a95010012",
    "ssh_key_id": 2
  }'
```

## Step 4: Install Unsloth and Dependencies

Connect to your instance and set up the environment:

```bash
# Clone the repository
git clone https://github.com/swan-alexchen/fine_tune_blog_demo.git
cd fine_tune_blog_demo

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Step 5: Load and Configure the Model

Think of this step as **preparing your workspace and tools** before starting a complex project. Just like a carpenter needs the right tools and workspace setup, we need to configure our model properly for efficient fine-tuning.

### Why We Need Model Configuration

**Memory Management**: Large language models are like massive libraries with billions of books (parameters). Without proper organization, they won't fit in your computer's memory. 4-bit quantization is like creating a compressed catalog system - it reduces memory usage by 75% while keeping the essential information intact.

```python
# Configuration - Setting up our "workspace"
max_seq_length = 2048 # How long conversations we can handle (like setting table size)
dtype = None # Auto-detect the best number format for your hardware
load_in_4bit = True # Compress the model to fit in memory (like zip files)
```

**Model Loading**: This is like hiring an experienced teacher who already knows a lot, rather than training someone from scratch:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

### Why LoRA (Low-Rank Adaptation)?

**The Surgery Metaphor**: Instead of performing major surgery on the entire brain (full fine-tuning), LoRA is like installing small, specialized implants that enhance specific capabilities. We're not changing the core personality, just adding new skills.

LoRA targets specific "attention" modules - think of these as the model's focus mechanisms, like how you pay attention to different parts of a conversation.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Size of our "skill implants" - bigger = more capacity, more memory
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",], # Which "brain regions" to enhance
    lora_alpha = 16, # How strongly to apply our new skills
    lora_dropout = 0, # Regularization (like practice variation)
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Memory optimization trick
    random_state = 3407, # Ensures reproducible results
)
```

## Step 6: Configure Chat Template and Dataset

### Why Chat Templates Matter

**The Translation Problem**: Imagine you're teaching someone to have conversations, but they only understand a very specific format. Chat templates are like teaching the model the "grammar rules" of conversations - who speaks when, how to indicate turns, and how to format responses properly.

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml", # The "conversation grammar" we're teaching
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # Translation dictionary
    map_eos_token = True, # How to say "conversation over"
)
```

### Why Dataset Formatting Is Crucial

**The Recipe Book Analogy**: Raw data is like having ingredients scattered everywhere. We need to organize them into a proper recipe book where each conversation follows the same format. This consistency helps the model learn patterns more effectively.

```python
from datasets import load_dataset

def formatting_prompts_func(examples):
    # Convert messy conversations into clean, formatted "recipes"
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

dataset = load_dataset("philschmid/guanaco-sharegpt-style", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

## Step 7: Training Configuration and Execution

### Why These Training Settings Matter

**The Learning Process**: Think of this like setting up a study schedule for a student. Too fast (high learning rate) and they get overwhelmed; too slow and they never improve. We need the right balance of challenge and support.

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Whether to combine short examples (like study groups)
    args = TrainingArguments(
        per_device_train_batch_size = 2, # How many examples to study at once
        gradient_accumulation_steps = 4, # Collect feedback before making changes
        warmup_steps = 5, # Gentle start, like warming up before exercise
        max_steps = 60, # Total number of learning iterations
        learning_rate = 2e-4, # How quickly to adapt (like study intensity)
        fp16 = not is_bfloat16_supported(), # Use efficient number formats
        bf16 = is_bfloat16_supported(),
        logging_steps = 1, # How often to check progress
        optim = "adamw_8bit", # The learning algorithm (like study method)
        weight_decay = 0.01, # Prevents over-memorization
        lr_scheduler_type = "linear", # Gradually reduce learning intensity
        seed = 3407, # Ensures reproducible results
        output_dir = "outputs",
        report_to = "none", # Where to log progress
    ),
)

trainer_stats = trainer.train() # Start the learning process!
```

**Key Training Concepts Explained**:
- **Batch Size**: Like study group size - too big and individual attention suffers, too small and learning is inefficient
- **Gradient Accumulation**: Like taking notes during multiple lectures before updating your understanding
- **Learning Rate**: The step size of improvement - like how much to adjust based on feedback
- **Warmup Steps**: Gentle introduction, like easing into a workout routine

## Step 8: Save the Fine-Tuned Model

### Why Saving Is Critical

**The Graduation Ceremony**: After all that training, we need to preserve what we've learned! This is like getting a diploma - proof of the new skills acquired.

```python
model.save_pretrained("blog_demo_lora")  # Save the new "skills" (LoRA adapters)
tokenizer.save_pretrained("blog_demo_lora") # Save the "language rules"
```

**What We're Actually Saving**:
- **LoRA Adapters**: The small "skill modules" we trained - these contain the new capabilities
- **Tokenizer**: The "dictionary" that translates between human language and model language
- **Configuration**: The "settings" that tell us how to use these components together

Think of it like saving a video game - you keep your character's new abilities and equipment, but the base game stays the same. This makes the saved model much smaller and easier to share than saving the entire massive model.


## Performance Comparison: Traditional vs Unsloth

| Metric | Traditional Method | Unsloth Method | Improvement |
|--------|-------------------|----------------|-------------|
| Training Speed | 1x | 2x | 100% faster |
| Memory Usage | 100% | 50% | 50% reduction |
| GPU Required (Llama3-8B) | A100 40GB | RTX 4090 24GB | Smaller GPU |
| Setup Complexity | High | Low | Much simpler |
| Accuracy | Baseline | Same* | No degradation |

*Unsloth maintains the same training quality while being significantly more efficient.

## Cost Optimization with Unsloth

### Monitor Your Credits
```bash
curl -X GET '{API_URL}/users/credits' \
  -H 'Authorization: Bearer {TOKEN/KEY}'
```

### Estimated Training Costs
With Unsloth's 2x speed improvement:
- **Llama3-8B on RTX 4090**: ~$5-10 for full fine-tune (vs $10-20 traditional)
- **Llama3-70B on A100**: ~$20-40 for full fine-tune (vs $40-80 traditional)

### Instance Management
```bash
# Stop when not training
curl -X GET '{API_URL}/computing/instance/{id}/stop' \
  -H 'Authorization: Bearer {TOKEN/KEY}'

# Check training progress remotely
curl -X GET '{API_URL}/computing/instance/{id}' \
  -H 'Authorization: Bearer {TOKEN/KEY}'
```


## Bringing Your Own Data for Custom Fine-Tuning

Now that you've seen how fine-tuning works with the demo dataset, let's explore how to prepare and use your own custom data to create domain-specific models tailored to your exact needs.

### Why Custom Data Matters

**From Generic to Specialized**: The demo showed you the mechanics, but custom data is where fine-tuning becomes powerful. Think of it like the difference between a general practitioner and a specialist doctor - both are valuable, but the specialist excels in their specific domain.

Custom datasets enable:
- **Domain Expertise**: Medical, legal, technical, or industry-specific knowledge
- **Brand Voice**: Consistent tone and style matching your organization
- **Specialized Tasks**: Customer support, code generation, content creation
- **Proprietary Knowledge**: Internal processes, products, or methodologies

### Choosing the Right Data Format

Different formats serve different purposes. Choose based on your end goal:

| Format | Best For | When to Use |
|--------|----------|-------------|
| **ShareGPT** | Conversational AI, chatbots | Building assistants, customer service |
| **Alpaca** | Task completion, instructions | Specific workflows, step-by-step guidance |
| **Raw Text** | Domain knowledge injection | Adding specialized vocabulary/concepts |

### ShareGPT Format: Building Conversational Experts

Perfect for creating domain-specific assistants:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "What's the difference between a 401k and a Roth IRA?"
    },
    {
      "from": "gpt",
      "value": "The key differences are: 1) Tax treatment - 401k contributions are pre-tax while Roth IRA contributions are post-tax..."
    }
  ]
}
```

**Real-World Example**: A financial advisory firm could create thousands of these Q&A pairs covering investment strategies, tax planning, and retirement advice.

### Alpaca Format: Teaching Specific Tasks

Ideal for instruction-following and task completion:

```json
{
  "instruction": "Generate a professional project status email",
  "input": "Project: Website redesign, Status: 75% complete, Issue: Delayed by client feedback",
  "output": "Subject: Website Redesign Project Update - Week of [Date]\n\nHi [Client Name],\n\nI wanted to provide you with an update on the website redesign project..."
}
```

**Real-World Example**: A marketing agency could train models to generate emails, proposals, and reports in their specific style and format.

### Data Collection Strategies

**1. Existing Content Mining**
- Customer support tickets and responses
- Documentation and FAQ sections
- Email templates and communications
- Training materials and guides

**2. Expert Knowledge Capture**
- Interview subject matter experts
- Record and transcribe training sessions
- Convert existing procedures to Q&A format
- Document decision-making processes

**3. Synthetic Data Generation**
- Use GPT-4 to generate domain-specific examples
- Create variations of existing high-quality examples
- Generate edge cases and challenging scenarios
- Expand limited datasets with AI assistance

### Dataset Preparation Workflow

**Step 1: Data Collection and Cleaning**
```python
import json
import pandas as pd

# Example: Converting CSV customer support data
df = pd.read_csv('support_tickets.csv')

conversations = []
for _, row in df.iterrows():
    conversation = {
        "conversations": [
            {"from": "human", "value": row['customer_question']},
            {"from": "gpt", "value": row['agent_response']}
        ]
    }
    conversations.append(conversation)

# Save as JSON
with open('custom_dataset.json', 'w') as f:
    json.dump(conversations, f, indent=2)
```

**Step 2: Quality Assurance**
- Remove personally identifiable information (PII)
- Check for factual accuracy
- Ensure consistent formatting
- Validate response quality
- Remove duplicates and near-duplicates

**Step 3: Dataset Validation**
```python
# Load and validate your dataset
from datasets import load_dataset

dataset = load_dataset("json", data_files="custom_dataset.json", split="train")

# Check dataset structure
print(f"Dataset size: {len(dataset)}")
print(f"Sample conversation: {dataset[0]}")

# Apply formatting (same as demo)
dataset = dataset.map(formatting_prompts_func, batched=True)
```

### Dataset Size Guidelines for Production

**Minimum Viable Dataset**: 500-1,000 examples
- Good for proof of concept
- Suitable for narrow, specific tasks
- Quick to create and iterate

**Production Quality**: 2,000-5,000 examples
- Covers most common scenarios
- Handles edge cases reasonably well
- Suitable for customer-facing applications

**Enterprise Grade**: 10,000+ examples
- Comprehensive coverage of domain
- Excellent handling of edge cases
- Professional-quality responses

### Advanced Data Techniques

**1. Data Augmentation**
```python
# Example: Creating variations of existing examples
original = "How do I reset my password?"
variations = [
    "I forgot my password, how can I reset it?",
    "What's the process for password recovery?",
    "Can you help me change my password?"
]
```

**2. Multi-turn Conversations**
```json
{
  "conversations": [
    {"from": "human", "value": "I need help with my account"},
    {"from": "gpt", "value": "I'd be happy to help! What specific issue are you experiencing?"},
    {"from": "human", "value": "I can't log in"},
    {"from": "gpt", "value": "Let's troubleshoot this. First, can you confirm the email address you're using?"}
  ]
}
```

**3. Context-Aware Training**
Include relevant context in your training data to improve model understanding of your domain.

### Loading Custom Datasets

Replace the demo dataset with your own:

```python
# Instead of the demo dataset
# dataset = load_dataset("philschmid/guanaco-sharegpt-style", split = "train")

# Use your custom dataset
dataset = load_dataset("json", data_files="your_custom_dataset.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
```

### Measuring Success with Custom Data

**Before/After Comparison**: Test your model on domain-specific questions before and after fine-tuning to measure improvement.

**Domain-Specific Evaluation**: Create a test set of questions specific to your use case and measure accuracy, relevance, and style consistency.

**User Feedback Loop**: Deploy in a controlled environment and gather real user feedback to identify areas for improvement.

### Common Pitfalls and Solutions

**Overfitting**: Too much repetitive data → Add diversity and variation
**Bias**: Unbalanced perspectives → Include multiple viewpoints and scenarios
**Inconsistency**: Mixed styles/formats → Standardize formatting and tone
**Insufficient Coverage**: Missing edge cases → Systematically identify and add corner cases

Your custom dataset is the foundation of a successful fine-tuned model. Invest time in quality data preparation - it's the difference between a generic AI and a specialized expert that truly understands your domain.

## Complete Working Example: Nebula Block API Assistant

Let's see the entire process in action with a real example. We'll create a specialized assistant that helps users interact with the Nebula Block API, trained on domain-specific Q&A data.

### The Complete Training Script

Here's the full [`bring_your_own_dataset.py`](bring_your_own_dataset.py:1) implementation that demonstrates the entire fine-tuning workflow:

```python
from unsloth import FastLanguageModel
import torch
from dotenv import load_dotenv
load_dotenv()

# Model configuration
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# Load the base model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# Configure LoRA for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# Set up chat template for conversation formatting
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)

# Dataset formatting function
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

# Load custom Nebula Block Q&A dataset
from qa_dataset import get_qa_dataset
dataset = get_qa_dataset()
dataset = dataset.map(formatting_prompts_func, batched = True,)

# Configure training parameters
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# Start training
trainer_stats = trainer.train()

# Save the fine-tuned model
model.save_pretrained("blog_demo_own_dataset_lora")  # Local saving
tokenizer.save_pretrained("blog_demo_own_dataset_lora")
```

### Training Output and Results

When you run the training script, you'll see output similar to this:

```
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
==((====))==  Unsloth 2025.6.8: Fast Mistral patching. Transformers: 4.53.0.
   \\   /|    NVIDIA H100 80GB HBM3. Num GPUs = 8. Max memory: 79.097 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.7.0+cu126. CUDA: 9.0. CUDA Toolkit: 12.6. Triton: 3.3.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.30. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth

Unsloth 2025.6.8 patched 32 layers with 32 QKV layers, 32 O layers and 32 MLP layers.
```

The training process completes in just 60 steps, demonstrating Unsloth's efficiency. The model learns to understand Nebula Block's API structure and provide accurate, helpful responses.

### Testing the Fine-Tuned Model

Here's the [`inference.py`](inference.py:1) script that loads and tests our fine-tuned model:

```python
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from dotenv import load_dotenv
load_dotenv()

max_seq_length = 20480 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# Load our fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "blog_demo_own_dataset_lora", # Our saved model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Configure LoRA (same settings as training)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# Set up chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
    map_eos_token = True,
)

# Enable fast inference
FastLanguageModel.for_inference(model)

# Test the model with a Nebula Block API question
messages = [
    {"from": "human", "value": "write me a python script to get all the ssh keys from nebula block:"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 1280, use_cache = True)
```

### Impressive Results: Domain-Specific Expertise

When we ask our fine-tuned model about Nebula Block's SSH keys API, it provides a perfect, production-ready response:

**Question**: "write me a python script to get all the ssh keys from nebula block:"

**Model Response**:
```python
import requests
import os

API_URL = "https://api.nebulablock.com/api/v1"
NEBULA_API_KEY = os.environ.get("NEBULA_API_KEY")

headers = {
   "Content-Type": "application/json",
   "Authorization": f"Bearer {NEBULA_API_KEY}"
}

response = requests.get(f'{API_URL}/ssh-keys', headers=headers)

if response.json().get("status") == "success":
   ssh_keys = response.json().get("data", [])
   print(f"Your SSH Keys:")
   for key in ssh_keys:
       print(f"  Name: {key['key_name']}, ID: {key['id']}, Public Key: {key['public_key']}")
else:
   print(f"Failed to retrieve SSH keys: {response.json().get('message')}")
```

### What Makes This Response Remarkable

**1. Perfect API Knowledge**: The model correctly uses:
- The exact Nebula Block API endpoint (`/ssh-keys`)
- Proper authentication headers with Bearer token
- Correct response structure parsing

**2. Production-Ready Code**: Includes:
- Environment variable usage for API keys
- Error handling for failed requests
- Clean, formatted output
- Proper JSON response parsing

**3. Domain-Specific Understanding**: The model learned:
- Nebula Block's specific API patterns
- Expected response formats
- Best practices for API interaction

This demonstrates how fine-tuning with domain-specific data creates models that understand your exact use case, API structure, and business context - something impossible with generic models.

## Conclusion

Combining Unsloth with Nebula Block GPU instances provides the most efficient and cost-effective way to fine-tune Llama3 models. With 2x faster training speeds and 50% memory reduction, you can achieve professional-quality results while minimizing costs and complexity.

Key benefits of this approach:
- **Faster Training**: Complete fine-tuning in half the time
- **Lower Costs**: Use smaller GPU instances and train faster
- **Better Results**: Maintain full accuracy with optimized training
- **Easy Scaling**: From 8B to 70B+ models on the same infrastructure
- **Production Ready**: Export to multiple formats for deployment

The combination of Unsloth's cutting-edge optimizations and Nebula Block's flexible GPU infrastructure makes fine-tuning large language models accessible to researchers, developers, and companies of all sizes.

For support and questions, visit the [Nebula Block contact page](https://nebulablock.com/contact) or explore the [customer portal](https://nebulablock.com/home) for additional resources.

---

*Ready to start fine-tuning with Unsloth? Sign up for Nebula Block at [https://nebulablock.com/register](https://nebulablock.com/register) and get $1 in free credits to begin your optimized Llama3 fine-tuning journey.*