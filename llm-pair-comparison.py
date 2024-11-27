import csv
import itertools
import math
import os
import re
import requests
import shutil
import textwrap
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import ollama


class ModelClient:
    """Abstract base class for model API clients."""
    def call(self, prompt: str, model: str) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt (str): Input prompt
            model (str): Model identifier
        
        Returns:
            str: Model's response
        """
        raise NotImplementedError("Subclasses must implement this method")


class OllamaClient(ModelClient):
    """Client for Ollama models."""
    def call(self, prompt: str, model: str) -> str:
        """
        Generate a response using Ollama.
        
        Args:
            prompt (str): Input prompt
            model (str): Ollama model identifier
        
        Returns:
            str: Model's response
        """
        try:
            response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Ollama API Error: {e}")
            return ""


class ModelComparisonEvaluator:
    """
    A comprehensive tool for comparing language models through interactive pairwise evaluation.
    """
    def __init__(self, terminal_width: Optional[int] = None):
        """
        Initialize the model comparison evaluator.
        
        Args:
            terminal_width (int, optional): Width for side-by-side display. 
                Defaults to current terminal width.
        """
        self.terminal_width = terminal_width or shutil.get_terminal_size().columns
        self.models = self._configure_model_clients()
        self.model_list = list(self.models.keys())
        self.rankings = {}
        self.pairwise_comparisons: List[Tuple[str, str, str]] = []

    def _configure_model_clients(self) -> Dict[str, ModelClient]:
        """
        Configure model clients with a flexible approach.
        
        Returns:
            Dict[str, ModelClient]: Mapping of model names to their respective clients
        """
        return {
            **{model: OllamaClient() for model in [
                "gemma2:2b",
                "gemma2:9b-instruct-q4_0",
                "granite3-dense:8b-instruct-q4_0",
                "llama3.1:8b-instruct-q4_0",
                "llama3.2:latest",
                "marco-o1:latest",
                "mistral:7b-instruct",
                "qwen2.5:14b-instruct-q2_K",
            ]}
        }

    def _format_side_by_side(self, text1: str, text2: str) -> str:
        """
        Format two texts side by side with equal width columns.
        
        Args:
            text1 (str): First text to display
            text2 (str): Second text to display
        
        Returns:
            str: Formatted side-by-side text
        """
        column_width = self.terminal_width // 2 - 4
        
        def wrap_text(text):
            lines = text.strip().split('\n')
            return '\n'.join(textwrap.fill(line, width=column_width) for line in lines)
        
        wrapped1 = wrap_text(text1)
        wrapped2 = wrap_text(text2)
        
        lines1 = wrapped1.split('\n')
        lines2 = wrapped2.split('\n')
        
        max_lines = max(len(lines1), len(lines2))
        lines1 += [''] * (max_lines - len(lines1))
        lines2 += [''] * (max_lines - len(lines2))
        
        column_width_with_padding = column_width + 2
        header = f"+{'-' * column_width_with_padding}+{'-' * column_width_with_padding}+"
        combined_lines = [
            header,
            f"| {'MODEL A':<{column_width}} | {'MODEL B':<{column_width}} |",
            header
        ]
        
        combined_lines.extend(
            f"| {l1:<{column_width}} | {l2:<{column_width}} |"
            for l1, l2 in zip(lines1, lines2)
        )
        combined_lines.append(header)
        
        return '\n'.join(combined_lines)

    def _generate_responses(self, model1: str, model2: str, prompt: str) -> List[Tuple[str, str, str, str]]:
        """
        Generate responses for two models.
        
        Args:
            model1 (str): First model name
            model2 (str): Second model name
            prompt (str): Input prompt
        
        Returns:
            List of tuples containing model names and their responses
        """
        try:
            response1 = self._clean_response(self.models[model1].call(prompt, model1))
            response2 = self._clean_response(self.models[model2].call(prompt, model2))
            return [(model1, model2, response1, response2)]
        except Exception as e:
            print(f"Error generating responses for {model1} and {model2}: {e}")
            return []

    @staticmethod
    def _clean_response(response: str) -> str:
        """
        Clean model responses by removing specific tags.
        
        Args:
            response (str): Raw model response
        
        Returns:
            str: Cleaned response
        """
        response = re.sub(r'<Thought>.*?</Thought>', '', response, flags=re.DOTALL)
        return re.sub(r'</?Output>', '', response).strip()

    def run_comprehensive_comparisons(self, prompt: str, num_rounds: int = 1):
        """
        Run comprehensive pairwise comparisons for all models.
        
        Args:
            prompt (str): Input prompt for model comparison
            num_rounds (int, optional): Number of comparison rounds. Defaults to 1.
        """
        print("Welcome to Local LLM Comparator!")
        print("You'll be shown responses from two models side by side.")
        print("Type 'A' for the left model, 'B' for the right model, or 'S' to skip.\n")
        
        model_pairs = list(itertools.combinations(self.model_list, 2))
        
        for i in range(0, len(model_pairs), 2):
            results = []
            threads = []
            comparison_pairs = []
            
            for j in range(2):
                if i + j < len(model_pairs):
                    model1, model2 = model_pairs[i + j]
                    comparison_pairs.append((model1, model2))
                    results.extend(self._generate_responses(model1, model2, prompt))
            
            for model1, model2, response1, response2 in results:
                print("\n" + self._format_side_by_side(response1, response2))
                self._process_user_comparison(model1, model2)

        self._compute_comprehensive_rankings()

    def _process_user_comparison(self, model1: str, model2: str):
        """
        Process user's model comparison choice.
        
        Args:
            model1 (str): First model name
            model2 (str): Second model name
        """
        while True:
            choice = input(f"\nWhich response is better? (A/B/S): ").strip().upper()
            
            if choice in ['A', 'B', 'S']:
                winner = model1 if choice == 'A' else (model2 if choice == 'B' else 'SKIP')
                log_message = (
                    f"{datetime.now()}: "
                    f"{winner if winner != 'SKIP' else 'Skipped'} "
                    f"{'was picked over' if winner not in ['SKIP'] else 'comparison between'} "
                    f"{model1} and {model2}\n"
                )
                
                self.pairwise_comparisons.append((model1, model2, winner))
                
                with open("comparison_log.txt", "a") as log_file:
                    log_file.write(log_message)
                
                break
            else:
                print("Invalid input. Please enter A, B, or S.")

            print("\n")

    def _compute_comprehensive_rankings(self):
        """
        Compute model rankings with a more sophisticated approach.
        """
        # Track direct wins and comparisons
        direct_wins = {model: {} for model in self.model_list}
        total_comparisons = {model: 0 for model in self.model_list}
        
        for model1, model2, winner in self.pairwise_comparisons:
            if winner == 'SKIP':
                continue
            
            if winner == model1:
                direct_wins[model1][model2] = direct_wins[model1].get(model2, 0) + 1
            elif winner == model2:
                direct_wins[model2][model1] = direct_wins[model2].get(model1, 0) + 1
            
            total_comparisons[model1] += 1
            total_comparisons[model2] += 1
        
        # Compute advanced ranking scores
        advanced_scores = {}
        for model in self.model_list:
            wins = sum(direct_wins[model].values())
            total_comps = total_comparisons[model]
            
            # Weighted score that considers both win rate and number of comparisons
            advanced_score = (wins / total_comps if total_comps > 0 else 0) * (1 + math.log(total_comps + 1))
            advanced_scores[model] = advanced_score
        
        # Sort by advanced score
        self.rankings = dict(sorted(advanced_scores.items(), key=lambda x: x[1], reverse=True))

    def _calculate_win_percentages(
        self, 
        model_scores: Dict[str, int], 
        model_comparisons: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Calculate win percentages for models.
        
        Args:
            model_scores (Dict[str, int]): Number of wins for each model
            model_comparisons (Dict[str, int]): Total number of comparisons for each model
        
        Returns:
            Dict[str, float]: Sorted dictionary of models and their win percentages
        """
        model_win_percentages = {
            model: (model_scores[model] / model_comparisons[model] * 100 
                    if model_comparisons[model] > 0 else 0)
            for model in self.model_list
        }
        
        return dict(sorted(model_win_percentages.items(), key=lambda x: x[1], reverse=True))

    def print_rankings(self):
        """
        Print and save final comprehensive rankings.
        """
        print("\n--- Final Model Rankings ---")
        for rank, (model, advanced_score) in enumerate(self.rankings.items(), 1):
            print(f"{rank}. {model}: {advanced_score:.4f} advanced score")
        
        filename = f"model_rankings_{int(time.time())}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Rank', 'Model', 'Advanced Score'])
            
            for rank, (model, advanced_score) in enumerate(self.rankings.items(), 1):
                csvwriter.writerow([rank, model, f"{advanced_score:.4f}"])
        
        print(f"\nRankings saved to {filename}")



def main():

    url = "https://r.jina.ai/https://careers.ua.edu/jobs/medical-billing-coder-525150-tuscaloosa-alabama-united-states"
    response = requests.get(url)
    page_content = response.text

    # You would pass in your specific prompt here
    prompt = f"""From the below JOB LISTING extract JOB TITLE, JOB SUMMARY, and JOB REQUIREMENTS, follow this format EXACTLY:
Job Title: [Single Line, No Formatting]

Job Summary: [Write EXACTLY 3-5 sentences describing role responsibilities, direct and factual]

Job Requirements: [Write EXACTLY 3-5 sentences covering qualifications, skills, and educational needs]


DO NOT EXPLAIN WHAT YOU ARE DOING, reply with EXACLTY the three requested sections!!! DO NOT use markdown, html, xml, or any other formatting! 


JOB LISTING:
{page_content}"""
    
    evaluator = ModelComparisonEvaluator()
    evaluator.run_comprehensive_comparisons(prompt, num_rounds=1)  # Increased rounds for more comprehensive comparison
    evaluator.print_rankings()

if __name__ == "__main__":
    main()