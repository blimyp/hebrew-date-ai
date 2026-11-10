using System.Globalization;
using System.Text;

namespace HebrewDateDataGenerator;

/// <summary>
/// Generates training data for a model that reads free-typed Hebrew dates.
/// Each row: the text a user might type, and the correct day, month and year.
/// </summary>
class Program
{
    const int VariantsPerDate = 3;        // how many different spellings per date
    const double TypoProbability = 0.15;  // share of rows with a deliberate typo

    static readonly Random Rng = new(42); // fixed seed: same data on every run

    // Canonical month labels, in HebrewCalendar order (month 1 = Tishrei).
    static readonly string[] RegularYearMonths =
        ["Tishrei", "Cheshvan", "Kislev", "Tevet", "Shevat", "Adar",
         "Nisan", "Iyar", "Sivan", "Tammuz", "Av", "Elul"];
    static readonly string[] LeapYearMonths =
        ["Tishrei", "Cheshvan", "Kislev", "Tevet", "Shevat", "AdarI", "AdarII",
         "Nisan", "Iyar", "Sivan", "Tammuz", "Av", "Elul"];

    // The ways people actually write each month.
    static readonly Dictionary<string, string[]> MonthSpellings = new()
    {
        ["Tishrei"]  = ["תשרי"],
        ["Cheshvan"] = ["חשון", "חשוון", "מרחשון", "מרחשוון"],
        ["Kislev"]   = ["כסלו", "כסליו"],
        ["Tevet"]    = ["טבת"],
        ["Shevat"]   = ["שבט"],
        ["Adar"]     = ["אדר"],
        ["AdarI"]    = ["אדר א", "אדר א׳", "אדר א'", "אדר ראשון"],
        ["AdarII"]   = ["אדר ב", "אדר ב׳", "אדר ב'", "אדר שני"],
        ["Nisan"]    = ["ניסן"],
        ["Iyar"]     = ["אייר", "איר"],
        ["Sivan"]    = ["סיון", "סיוון"],
        ["Tammuz"]   = ["תמוז"],
        ["Av"]       = ["אב", "מנחם אב"],
        ["Elul"]     = ["אלול"],
    };

    static void Main()
    {
        var calendar = new HebrewCalendar();
        var start = new DateTime(1600, 1, 1);
        var end = new DateTime(2200, 12, 31);

        Directory.CreateDirectory("data");
        using var writer = new StreamWriter("data/hebrew_dates.csv", false, new UTF8Encoding(true));
        // The year is also split into its three letter-digits (תשפ״ז = 7, 8, 7), so the trainer
        // can learn each digit with a small model instead of one big model with ~290 years.
        writer.WriteLine("Text,Day,Month,Year,YearHundreds,YearTens,YearUnits");

        int rows = 0;
        for (var date = start; date <= end; date = date.AddDays(1))
        {
            int year = calendar.GetYear(date);
            int month = calendar.GetMonth(date);
            int day = calendar.GetDayOfMonth(date);
            string monthLabel = calendar.IsLeapYear(year)
                ? LeapYearMonths[month - 1]
                : RegularYearMonths[month - 1];

            for (int i = 0; i < VariantsPerDate; i++)
            {
                var (text, hasYear) = WriteDate(day, monthLabel, year);

                // Year 0 means the user didn't type a year; then all three digits are 0 too.
                // Hundreds is never 0 in a real year (תק = 5 ... תת = 8), so it marks "no year".
                int typedYear = hasYear ? year : 0;
                int hundreds = hasYear ? year % 1000 / 100 : 0;
                int tens = hasYear ? year % 100 / 10 : 0;
                int units = hasYear ? year % 10 : 0;
                writer.WriteLine($"{Csv(text)},{day},{monthLabel},{typedYear},{hundreds},{tens},{units}");
                rows++;
            }
        }

        Console.WriteLine($"Wrote {rows:N0} rows to data/hebrew_dates.csv");
    }

    /// <summary>Writes one date the way a person might type it.</summary>
    static (string Text, bool HasYear) WriteDate(int day, string monthLabel, int year)
    {
        int style = Rng.Next(3); // 0 = ׳ ״, 1 = ' ", 2 = no punctuation
        var spellings = MonthSpellings[monthLabel];
        string month = spellings[Rng.Next(spellings.Length)];
        // Typos go only in the month name, so the day and year stay readable.
        if (Rng.NextDouble() < TypoProbability) month = AddTypo(month);

        string dayText = Punctuate(ToHebrewLetters(day), style);
        string prefix = Rng.NextDouble() < 0.5 ? "ב" : "";   // "ט״ו בשבט" or "ט״ו שבט"
        string text = $"{dayText} {prefix}{month}";

        int yearMode = Rng.Next(3); // 0 = no year, 1 = תשפ״ז, 2 = ה׳תשפ״ז
        if (yearMode == 0) return (text, false);

        string yearLetters = ToHebrewLetters(year % 1000);
        if (Rng.NextDouble() < 0.5) yearLetters = UseFinalLetter(yearLetters); // תש״ף
        string yearText = Punctuate(yearLetters, style);
        if (yearMode == 2) yearText = Punctuate("ה", style) + yearText;

        return ($"{text} {yearText}", true);
    }

    /// <summary>Converts a number (1–999) to Hebrew letters, e.g. 15 → טו, 787 → תשפז.</summary>
    static string ToHebrewLetters(int n)
    {
        var sb = new StringBuilder();
        while (n >= 400) { sb.Append('ת'); n -= 400; }
        if (n >= 100) { sb.Append("קרש"[n / 100 - 1]); n %= 100; }
        if (n == 15) return sb.Append("טו").ToString();   // not יה
        if (n == 16) return sb.Append("טז").ToString();   // not יו
        if (n >= 10) { sb.Append("יכלמנסעפצ"[n / 10 - 1]); n %= 10; }
        if (n > 0) sb.Append("אבגדהוזחט"[n - 1]);
        return sb.ToString();
    }

    /// <summary>Adds a geresh to one letter, or gershayim before the last of several.</summary>
    static string Punctuate(string letters, int style)
    {
        if (style == 2) return letters;
        string geresh = style == 0 ? "׳" : "'";
        string gershayim = style == 0 ? "״" : "\"";
        return letters.Length == 1
            ? letters + geresh
            : letters[..^1] + gershayim + letters[^1];
    }

    /// <summary>Swaps a final-position letter to its final form: תשפ → תשף.</summary>
    static string UseFinalLetter(string letters)
    {
        const string regular = "כמנפצ", final = "ךםןףץ";
        int i = regular.IndexOf(letters[^1]);
        return i < 0 ? letters : letters[..^1] + final[i];
    }

    /// <summary>Adds one realistic typo: a missing, doubled or swapped letter.</summary>
    static string AddTypo(string text)
    {
        var letters = Enumerable.Range(0, text.Length)
            .Where(i => text[i] >= 'א' && text[i] <= 'ת').ToList();
        if (letters.Count < 3) return text; // too short: a typo would change the month
        int pos = letters[Rng.Next(letters.Count)];

        return Rng.Next(3) switch
        {
            0 => text.Remove(pos, 1),
            1 => text.Insert(pos, text[pos].ToString()),
            _ => pos + 1 < text.Length && text[pos + 1] != ' '
                ? text[..pos] + text[pos + 1] + text[pos] + text[(pos + 2)..]
                : text,
        };
    }

    /// <summary>Quotes a CSV field, since Hebrew dates can contain " and '.</summary>
    static string Csv(string field) => "\"" + field.Replace("\"", "\"\"") + "\"";
}
