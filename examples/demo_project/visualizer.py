"""
Data Visualization Module

This module provides functionality for creating visualizations from data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for compatibility


class Visualizer:
    """A class for creating data visualizations."""

    def __init__(self, data: Dict[str, Any]):
        """Initialize the visualizer with data."""
        self.data = data

    def create_bar_chart(self, title: str = "Data Summary",
                        xlabel: str = "Category",
                        ylabel: str = "Value",
                        save_path: Optional[str] = None) -> None:
        """Create a bar chart from the data."""
        if not self.data:
            print("No data to visualize")
            return

        labels = list(self.data.keys())
        values = list(self.data.values())

        plt.figure(figsize=(10, 6))
        sns.barplot(x=labels, y=values)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Chart saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def create_pie_chart(self, title: str = "Data Distribution",
                        save_path: Optional[str] = None) -> None:
        """Create a pie chart from the data."""
        if not self.data:
            print("No data to visualize")
            return

        labels = list(self.data.keys())
        values = list(self.data.values())

        plt.figure(figsize=(8, 8))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(title)
        plt.axis('equal')

        if save_path:
            plt.savefig(save_path)
            print(f"Pie chart saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def create_line_chart(self, title: str = "Data Trend",
                         xlabel: str = "X",
                         ylabel: str = "Y",
                         save_path: Optional[str] = None) -> None:
        """Create a line chart from the data."""
        if not self.data:
            print("No data to visualize")
            return

        x_values = list(self.data.keys())
        y_values = list(self.data.values())

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker='o')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Line chart saved to {save_path}")
        else:
            plt.show()

        plt.close()