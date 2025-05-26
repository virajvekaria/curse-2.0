"""
Data Analysis Demo

This demo shows how to use the modular data analysis components together.
"""

from csv_processor import CSVProcessor
from data_analyzer import DataAnalyzer
from visualizer import Visualizer


def main():
    """Main demo function showing the data analysis workflow."""
    try:
        # Step 1: Load data from CSV
        print("📊 Loading data from CSV...")
        import os
        # Use the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'data.csv')
        processor = CSVProcessor(csv_path)
        data = processor.load_data()

        if not data:
            print("❌ No data loaded. Please check the CSV file.")
            return

        print(f"✅ Loaded {len(data)} rows of data")
        print(f"📋 Headers: {processor.get_headers()}")

        # Step 2: Analyze the data
        print("\n🔍 Analyzing data...")
        analyzer = DataAnalyzer(data)
        summary = analyzer.generate_summary()
        statistics = analyzer.calculate_statistics()

        print(f"📈 Data Summary: {summary}")
        if statistics:
            print(f"📊 Statistics: {statistics}")

        # Step 3: Create visualizations
        print("\n📊 Creating visualizations...")
        if summary:
            visualizer = Visualizer(summary)

            # Create different types of charts
            visualizer.create_bar_chart(
                title="Data Analysis Results",
                xlabel="Categories",
                ylabel="Values",
                save_path="data_bar_chart.png"
            )

            visualizer.create_pie_chart(
                title="Data Distribution",
                save_path="data_pie_chart.png"
            )

            print("✅ Visualizations created and saved!")
        else:
            print("⚠️ No summary data available for visualization")

    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()