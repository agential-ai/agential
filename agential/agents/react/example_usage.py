"""Example usage of the ScalableReAct agent with improved logging."""

from agential.agents.react.scalable_agent import ScalableReAct
from agential.core.llm import MockLLM  # For demonstration

def main():
    """Demonstrate the improved logging and metrics system."""
    
    # Create a mock LLM for demonstration
    llm = MockLLM()
    
    # Create agent with verbose logging and rich output
    agent = ScalableReAct(
        llm=llm,
        benchmark="gsm8k",
        verbose=True,  # Enable logging
        use_rich=True,  # Use colorful output (if rich is available)
        max_steps=3
    )
    
    # Example question
    question = "Janet's dogs eat 2 pounds of food each day. How many pounds of food do they eat in a week?"
    
    print("🚀 Running ReAct Agent with improved logging...")
    print("=" * 80)
    
    # Generate answer with full logging
    result = agent.generate(question)
    
    print("\n" + "=" * 80)
    print("📈 DETAILED METRICS ACCESS")
    print("=" * 80)
    
    # Access detailed metrics
    metrics = agent.get_metrics()
    
    print(f"📊 Execution Summary:")
    print(f"   Start Time: {metrics.start_time}")
    print(f"   End Time: {metrics.end_time}")
    print(f"   Total Steps: {metrics.total_steps}")
    print(f"   Total Tokens: {metrics.total_tokens:,}")
    print(f"   Total Cost: ${metrics.total_cost:.4f}")
    print(f"   Total Time: {metrics.total_time:.2f}s")
    
    print(f"\n📋 Step-by-Step Breakdown:")
    for step in metrics.steps:
        print(f"   Step {step.step_number}:")
        print(f"     Action: {step.action_type}")
        print(f"     Tokens: {step.total_tokens}")
        print(f"     Cost: ${step.total_cost:.4f}")
        print(f"     Time: {step.total_time:.2f}s")
        print(f"     Finished: {step.finished}")
    
    # Convert to dictionary for serialization
    metrics_dict = metrics.to_dict()
    print(f"\n💾 Serialized Metrics Keys: {list(metrics_dict.keys())}")
    
    print(f"\n🎯 Final Answer: {result.answer}")
    
    # Example with silent mode
    print("\n" + "=" * 80)
    print("🔇 SILENT MODE EXAMPLE")
    print("=" * 80)
    
    silent_agent = ScalableReAct(
        llm=llm,
        benchmark="gsm8k",
        verbose=False,  # Disable logging
        max_steps=2
    )
    
    silent_result = silent_agent.generate("What is 5 + 3?")
    silent_metrics = silent_agent.get_metrics()
    
    print(f"Silent execution completed!")
    print(f"Answer: {silent_result.answer}")
    print(f"Total tokens: {silent_metrics.total_tokens}")
    print(f"Total cost: ${silent_metrics.total_cost:.4f}")

if __name__ == "__main__":
    main() 