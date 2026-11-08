using System.Globalization;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace HebrewDateTrainer;

/// <summary>One row of the CSV the data generator wrote.</summary>
public class DateRow
{
    [LoadColumn(0)] public string Text { get; set; } = "";
    [LoadColumn(1)] public string Day { get; set; } = "";
    [LoadColumn(2)] public string Month { get; set; } = "";
    [LoadColumn(3)] public string Year { get; set; } = "";
}

/// <summary>What a model answers: its guess and how sure it is of each option.</summary>
public class Guess
{
    public string PredictedValue { get; set; } = "";
    public float[] Score { get; set; } = [];
}

class Program
{
    static readonly string DataPath = Path.Combine("..", "create_examples", "data", "hebrew_dates.csv");
    const string ModelsFolder = "models";

    static readonly string[] RegularYearMonths =
        ["Tishrei", "Cheshvan", "Kislev", "Tevet", "Shevat", "Adar",
         "Nisan", "Iyar", "Sivan", "Tammuz", "Av", "Elul"];
    static readonly string[] LeapYearMonths =
        ["Tishrei", "Cheshvan", "Kislev", "Tevet", "Shevat", "AdarI", "AdarII",
         "Nisan", "Iyar", "Sivan", "Tammuz", "Av", "Elul"];

    static void Main()
    {
        Console.InputEncoding = Encoding.UTF8;
        Console.OutputEncoding = Encoding.UTF8;

        if (!File.Exists(DataPath))
        {
            Console.WriteLine($"Training data not found at: {Path.GetFullPath(DataPath)}");
            Console.WriteLine("Run the data generator first (dotnet run in its folder).");
            return;
        }

        var ml = new MLContext(seed: 1);
        IDataView data = ml.Data.LoadFromTextFile<DateRow>(
            DataPath, separatorChar: ',', hasHeader: true, allowQuoting: true);

        // Three small models, one per part of the date.
        var dayModel = LoadOrTrain(ml, data, "Day");
        var monthModel = LoadOrTrain(ml, data, "Month");
        var yearModel = LoadOrTrain(ml, data, "Year");

        TryItYourself(ml, dayModel, monthModel, yearModel);
    }

    /// <summary>Loads a saved model, or trains, tests and saves a new one.</summary>
    static ITransformer LoadOrTrain(MLContext ml, IDataView data, string column)
    {
        string path = Path.Combine(ModelsFolder, $"{column.ToLower()}.zip");
        if (File.Exists(path))
        {
            Console.WriteLine($"Loaded the {column} model from {path}");
            return ml.Model.Load(path, out _);
        }

        // Keep 20% of the examples aside, so we test on rows the model never saw.
        var split = ml.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);

        var pipeline = ml.Transforms.Conversion.MapValueToKey("Label", column)
            // Turn the text into numbers: short letter sequences (e.g. "שבט", "״ו")
            // and words and word pairs, so the model sees both spelling and order.
            .Append(ml.Transforms.Text.FeaturizeText("Features", new TextFeaturizingEstimator.Options
            {
                CharFeatureExtractor = new WordBagEstimator.Options { NgramLength = 3, UseAllLengths = true },
                WordFeatureExtractor = new WordBagEstimator.Options { NgramLength = 2, UseAllLengths = true },
            }, nameof(DateRow.Text)))
            .Append(ml.MulticlassClassification.Trainers.LbfgsMaximumEntropy())
            .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedValue", "PredictedLabel"));

        Console.WriteLine($"Training the {column} model... (this can take a few minutes)");
        var watch = System.Diagnostics.Stopwatch.StartNew();
        var model = pipeline.Fit(split.TrainSet);

        var metrics = ml.MulticlassClassification.Evaluate(model.Transform(split.TestSet));
        Console.WriteLine($"  {column}: {metrics.MicroAccuracy:P2} correct on unseen examples " +
                          $"({watch.Elapsed.TotalSeconds:N0} s)");

        Directory.CreateDirectory(ModelsFolder);
        ml.Model.Save(model, data.Schema, path);
        return model;
    }

    /// <summary>Lets you type dates and see what the models make of them.</summary>
    static void TryItYourself(MLContext ml, ITransformer day, ITransformer month, ITransformer year)
    {
        var dayEngine = ml.Model.CreatePredictionEngine<DateRow, Guess>(day);
        var monthEngine = ml.Model.CreatePredictionEngine<DateRow, Guess>(month);
        var yearEngine = ml.Model.CreatePredictionEngine<DateRow, Guess>(year);

        Console.WriteLine();
        Console.WriteLine("Type a Hebrew date and press Enter (an empty line exits).");
        Console.WriteLine("Note: the terminal may show Hebrew letters in reverse order; that's only the display.");

        while (true)
        {
            Console.Write("> ");
            string? text = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(text)) break;

            var input = new DateRow { Text = text };
            var d = dayEngine.Predict(input);
            var m = monthEngine.Predict(input);
            var y = yearEngine.Predict(input);

            string yearText = y.PredictedValue == "0" ? "not typed" : y.PredictedValue;
            Console.WriteLine($"  Day: {d.PredictedValue} ({Sure(d):P0})  " +
                              $"Month: {m.PredictedValue} ({Sure(m):P0})  " +
                              $"Year: {yearText} ({Sure(y):P0})");
            Console.WriteLine($"  Gregorian: {ToGregorian(d.PredictedValue, m.PredictedValue, y.PredictedValue)}");
        }
    }

    /// <summary>How sure the model is of its own answer (0–100%).</summary>
    static float Sure(Guess g) => g.Score.Length == 0 ? 0 : g.Score.Max();

    /// <summary>Combines the three answers into a Gregorian date, if they make a real date.</summary>
    static string ToGregorian(string dayText, string monthLabel, string yearText)
    {
        if (yearText == "0") return "(no year typed)";
        if (!int.TryParse(dayText, out int day) || !int.TryParse(yearText, out int year))
            return "(could not read the answer)";

        var calendar = new HebrewCalendar();
        var months = calendar.IsLeapYear(year) ? LeapYearMonths : RegularYearMonths;
        int month = Array.IndexOf(months, monthLabel) + 1;
        if (month == 0) return $"({monthLabel} does not exist in {year})";

        try
        {
            return calendar.ToDateTime(year, month, day, 0, 0, 0, 0).ToString("dd/MM/yyyy");
        }
        catch (ArgumentOutOfRangeException)
        {
            return $"(day {day} does not exist in {monthLabel} {year})";
        }
    }
}
