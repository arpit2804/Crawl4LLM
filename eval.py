import os
import re
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse

# Import the required modules from the project
from corpus_interface import ClueWeb22Api, UnifiedGetter
from document_rater import DocumentRater, DocumentLengthRater

class DocumentAnalyzer:
    """Class to analyze documents retrieved from different crawling strategies"""
    
    def __init__(self, cw22_root_path, crawl_results_dir):
        """Initialize the analyzer with paths and APIs"""
        self.cw22_root_path = cw22_root_path
        self.crawl_results_dir = crawl_results_dir
        self.cw22_api = ClueWeb22Api(cw22_root_path)
        self.unified_getter = UnifiedGetter(self.cw22_api)
        
        # Initialize NLTK components
        try:
            import nltk
            nltk.download('punkt', quiet=True)
        except ImportError:
            print("NLTK not installed. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
            import nltk
            nltk.download('punkt', quiet=True)
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # BLEU smoothing function
        self.smoothing = SmoothingFunction().method1
        
        # Store documents by strategy
        self.documents_by_strategy = {}
        self.docids_by_strategy = {}

    def get_strategy_folders(self):
        """Get all strategy folders in the crawl results directory"""
        return [d for d in os.listdir(self.crawl_results_dir) 
                if os.path.isdir(os.path.join(self.crawl_results_dir, d)) 
                and d.startswith("seed_10_crawl_200")]

    def read_docids_from_strategy(self, strategy_name):
        """Read all document IDs from a strategy folder"""
        strategy_dir = os.path.join(self.crawl_results_dir, strategy_name)
        docids = []
        
        # Find all iteration files and sort them
        iter_files = sorted([f for f in os.listdir(strategy_dir) if f.startswith("iter_") and f.endswith(".docids.txt")])
        
        for iter_file in iter_files:
            with open(os.path.join(strategy_dir, iter_file), 'r') as f:
                docids.extend([line.strip() for line in f])
                
        return docids

    def fetch_document_text(self, docid):
        """Fetch the text of a document by ID"""
        doc = self.unified_getter.get_doc(docid)
        if doc and doc.text:
            return doc.text
        return None

    def fetch_documents_for_strategy(self, strategy_name, max_workers=10):
        """Fetch all documents for a strategy using parallel processing"""
        print(f"Fetching documents for strategy: {strategy_name}")
        docids = self.read_docids_from_strategy(strategy_name)
        self.docids_by_strategy[strategy_name] = docids
        
        # Use multiprocessing to fetch documents in parallel
        with Pool(max_workers) as pool:
            texts = list(tqdm(
                pool.imap(self.fetch_document_text, docids),
                total=len(docids),
                desc=f"Fetching {strategy_name}"
            ))
        
        # Filter out None values
        documents = [(docid, text) for docid, text in zip(docids, texts) if text is not None]
        print(f"Retrieved {len(documents)}/{len(docids)} documents for {strategy_name}")
        
        self.documents_by_strategy[strategy_name] = documents
        return documents

    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        if not text:
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def tokenize(self, text):
        """Tokenize text into words"""
        if not text:
            return []
        try:
            import nltk
            return nltk.word_tokenize(text)
        except:
            # Fallback simple tokenization
            return text.split()

    def calculate_rouge_l(self, text1, text2):
        """Calculate ROUGE-L score between two texts"""
        if not text1 or not text2:
            return 0.0
        scores = self.rouge_scorer.score(text1, text2)
        return scores['rougeL'].fmeasure

    def calculate_bleu(self, text1, text2):
        """Calculate BLEU score between two texts"""
        if not text1 or not text2:
            return 0.0
        
        reference = [self.tokenize(text1)]
        hypothesis = self.tokenize(text2)
        
        # Handle empty sequences
        if not reference[0] or not hypothesis:
            return 0.0
            
        # Calculate BLEU with smoothing
        try:
            return sentence_bleu(reference, hypothesis, smoothing_function=self.smoothing)
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            return 0.0

    def calculate_jaccard(self, text1, text2):
        """Calculate Jaccard similarity between two texts"""
        if not text1 or not text2:
            return 0.0
            
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0

    def calculate_tfidf_similarity(self, texts):
        """Calculate TF-IDF cosine similarity matrix for a list of texts"""
        if not texts or all(not t for t in texts):
            return np.zeros((len(texts), len(texts)))
            
        # Filter out None values and empty strings
        valid_texts = [t for t in texts if t]
        if not valid_texts:
            return np.zeros((len(texts), len(texts)))
            
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(analyzer='word', lowercase=True, stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(valid_texts)
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
        except Exception as e:
            print(f"TF-IDF calculation error: {e}")
            return np.zeros((len(valid_texts), len(valid_texts)))

    def analyze_strategy(self, strategy_name):
        """Analyze documents from a single strategy"""
        if strategy_name not in self.documents_by_strategy:
            print(f"Documents for {strategy_name} not loaded. Loading now...")
            self.fetch_documents_for_strategy(strategy_name)
        
        documents = self.documents_by_strategy[strategy_name]
        _, texts = zip(*documents) if documents else ([], [])
        
        # Preprocess texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        results = {
            'num_documents': len(documents),
            'avg_document_length': np.mean([len(text) for text in texts]) if texts else 0,
            'median_document_length': np.median([len(text) for text in texts]) if texts else 0,
        }
        
        # Calculate ROUGE-L scores (each doc against all others)
        rouge_scores = []
        for i, text1 in enumerate(preprocessed_texts):
            for j, text2 in enumerate(preprocessed_texts):
                if i < j:  # Only compare each pair once
                    score = self.calculate_rouge_l(text1, text2)
                    rouge_scores.append(score)
        
        results['rouge_l_mean'] = np.mean(rouge_scores) if rouge_scores else 0
        results['rouge_l_median'] = np.median(rouge_scores) if rouge_scores else 0
        results['rouge_l_std'] = np.std(rouge_scores) if rouge_scores else 0
        
        # Calculate BLEU scores
        bleu_scores = []
        for i, text1 in enumerate(preprocessed_texts):
            for j, text2 in enumerate(preprocessed_texts):
                if i < j:  # Only compare each pair once
                    score = self.calculate_bleu(text1, text2)
                    bleu_scores.append(score)
        
        results['bleu_mean'] = np.mean(bleu_scores) if bleu_scores else 0
        results['bleu_median'] = np.median(bleu_scores) if bleu_scores else 0
        results['bleu_std'] = np.std(bleu_scores) if bleu_scores else 0
        
        # Calculate Jaccard similarities
        jaccard_scores = []
        for i, text1 in enumerate(preprocessed_texts):
            for j, text2 in enumerate(preprocessed_texts):
                if i < j:  # Only compare each pair once
                    score = self.calculate_jaccard(text1, text2)
                    jaccard_scores.append(score)
        
        results['jaccard_mean'] = np.mean(jaccard_scores) if jaccard_scores else 0
        results['jaccard_median'] = np.median(jaccard_scores) if jaccard_scores else 0
        results['jaccard_std'] = np.std(jaccard_scores) if jaccard_scores else 0
        
        # Calculate TF-IDF similarities
        if preprocessed_texts:
            tfidf_matrix = self.calculate_tfidf_similarity(preprocessed_texts)
            tfidf_scores = []
            for i in range(len(tfidf_matrix)):
                for j in range(len(tfidf_matrix)):
                    if i < j:  # Only include each pair once
                        tfidf_scores.append(tfidf_matrix[i, j])
            
            results['tfidf_mean'] = np.mean(tfidf_scores) if tfidf_scores else 0
            results['tfidf_median'] = np.median(tfidf_scores) if tfidf_scores else 0
            results['tfidf_std'] = np.std(tfidf_scores) if tfidf_scores else 0
        else:
            results['tfidf_mean'] = 0
            results['tfidf_median'] = 0
            results['tfidf_std'] = 0
        
        return results

    def analyze_all_strategies(self):
        """Analyze all strategies and compare them"""
        strategies = self.get_strategy_folders()
        all_results = {}
        
        for strategy in strategies:
            print(f"Analyzing strategy: {strategy}")
            results = self.analyze_strategy(strategy)
            all_results[strategy] = results
            
        return all_results

    def generate_comparison_table(self, all_results):
        """Generate a comparison table of all strategies"""
        if not all_results:
            return None
            
        # Convert results to DataFrame
        df = pd.DataFrame(all_results).T
        
        # Reorder columns for better readability
        column_order = [
            'num_documents', 'avg_document_length', 'median_document_length',
            'rouge_l_mean', 'rouge_l_median', 'rouge_l_std',
            'bleu_mean', 'bleu_median', 'bleu_std',
            'jaccard_mean', 'jaccard_median', 'jaccard_std',
            'tfidf_mean', 'tfidf_median', 'tfidf_std'
        ]
        
        df = df[column_order]
        return df

    def plot_comparison(self, df, output_dir):
        """Generate comparison plots and save them"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Bar plot of mean metric values across strategies
        plt.figure(figsize=(12, 8))
        mean_metrics = ['rouge_l_mean', 'bleu_mean', 'jaccard_mean', 'tfidf_mean']
        df_means = df[mean_metrics]
        
        # Normalize values for better comparison
        df_plot = df_means.copy()
        for col in df_means.columns:
            max_val = df_means[col].max()
            if max_val > 0:
                df_plot[col] = df_means[col] / max_val
        
        ax = df_plot.plot(kind='bar', figsize=(12, 8))
        plt.title('Normalized Mean Metrics by Crawling Strategy')
        plt.ylabel('Normalized Score')
        plt.xlabel('Strategy')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mean_metrics_comparison.png'))
        plt.close()
        
        # 2. Heatmap showing all metrics
        plt.figure(figsize=(14, 10))
        sns.heatmap(df, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Metrics Heatmap by Crawling Strategy')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'))
        plt.close()
        
        # 3. Individual metric comparisons
        metrics_to_plot = [
            ('rouge_l_mean', 'ROUGE-L Mean'),
            ('bleu_mean', 'BLEU Mean'),
            ('jaccard_mean', 'Jaccard Similarity Mean'),
            ('tfidf_mean', 'TF-IDF Cosine Similarity Mean')
        ]
        
        for metric, title in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=df.index, y=df[metric])
            plt.title(title)
            plt.ylabel('Score')
            plt.xlabel('Strategy')
            plt.xticks(rotation=45, ha='right')
            
            # Add values on top of bars
            for i, v in enumerate(df[metric]):
                ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
            plt.close()
        
        # 4. Document length comparison
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=df.index, y=df['avg_document_length'])
        plt.title('Average Document Length by Strategy')
        plt.ylabel('Characters')
        plt.xlabel('Strategy')
        plt.xticks(rotation=45, ha='right')
        
        # Add values on top of bars
        for i, v in enumerate(df['avg_document_length']):
            ax.text(i, v + 0.01, f"{v:.0f}", ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'avg_doc_length_comparison.png'))
        plt.close()

    def save_results(self, df, output_dir):
        """Save results to CSV and JSON files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        df.to_csv(os.path.join(output_dir, 'strategy_comparison_results.csv'))
        
        # Save as JSON
        with open(os.path.join(output_dir, 'strategy_comparison_results.json'), 'w') as f:
            f.write(df.to_json(orient='index', indent=2))
        
        # Save summary text file
        with open(os.path.join(output_dir, 'strategy_comparison_summary.txt'), 'w') as f:
            f.write("Crawling Strategy Comparison Summary\n")
            f.write("===================================\n\n")
            
            # Add summary for each strategy
            for strategy in df.index:
                f.write(f"Strategy: {strategy}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Number of documents: {df.loc[strategy, 'num_documents']:.0f}\n")
                f.write(f"Average document length: {df.loc[strategy, 'avg_document_length']:.0f} characters\n")
                f.write(f"Median document length: {df.loc[strategy, 'median_document_length']:.0f} characters\n")
                f.write("\nSimilarity Metrics:\n")
                f.write(f"  ROUGE-L: {df.loc[strategy, 'rouge_l_mean']:.4f} (mean), {df.loc[strategy, 'rouge_l_median']:.4f} (median)\n")
                f.write(f"  BLEU: {df.loc[strategy, 'bleu_mean']:.4f} (mean), {df.loc[strategy, 'bleu_median']:.4f} (median)\n")
                f.write(f"  Jaccard: {df.loc[strategy, 'jaccard_mean']:.4f} (mean), {df.loc[strategy, 'jaccard_median']:.4f} (median)\n")
                f.write(f"  TF-IDF Cosine: {df.loc[strategy, 'tfidf_mean']:.4f} (mean), {df.loc[strategy, 'tfidf_median']:.4f} (median)\n")
                f.write("\n\n")
            
            # Add comparative analysis
            f.write("Comparative Analysis\n")
            f.write("===================\n\n")
            
            # Find best strategy for each metric
            metrics = {
                'ROUGE-L': 'rouge_l_mean',
                'BLEU': 'bleu_mean',
                'Jaccard': 'jaccard_mean',
                'TF-IDF Cosine': 'tfidf_mean'
            }
            
            for metric_name, column in metrics.items():
                best_strategy = df[column].idxmax()
                worst_strategy = df[column].idxmin()
                f.write(f"{metric_name} similarity:\n")
                f.write(f"  Highest: {best_strategy} ({df.loc[best_strategy, column]:.4f})\n")
                f.write(f"  Lowest: {worst_strategy} ({df.loc[worst_strategy, column]:.4f})\n")
                f.write(f"  Ratio (highest/lowest): {df.loc[best_strategy, column] / df.loc[worst_strategy, column]:.2f}x\n\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze and compare document crawling strategies')
    parser.add_argument('--cw22_root_path', type=str, required=True, help='Path to ClueWeb22 root directory')
    parser.add_argument('--crawl_results_dir', type=str, required=True, help='Path to crawl results directory')
    parser.add_argument('--output_dir', type=str, default='strategy_comparison_results', help='Directory to save results')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of worker processes')
    
    args = parser.parse_args()
    
    print(f"Initializing analysis with ClueWeb22 root: {args.cw22_root_path}")
    analyzer = DocumentAnalyzer(args.cw22_root_path, args.crawl_results_dir)
    
    # Fetch documents for all strategies
    strategies = analyzer.get_strategy_folders()
    print(f"Found {len(strategies)} crawling strategies: {strategies}")
    
    for strategy in strategies:
        analyzer.fetch_documents_for_strategy(strategy, max_workers=args.max_workers)
    
    # Analyze all strategies
    results = analyzer.analyze_all_strategies()
    
    # Generate comparison table
    comparison_df = analyzer.generate_comparison_table(results)
    print("\nComparison Results:")
    print(comparison_df)
    
    # Create visualizations
    analyzer.plot_comparison(comparison_df, args.output_dir)
    
    # Save results
    analyzer.save_results(comparison_df, args.output_dir)
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()