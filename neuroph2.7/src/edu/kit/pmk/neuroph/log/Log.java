package edu.kit.pmk.neuroph.log;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;

import edu.kit.pmk.neuroph.eval.Score;
import edu.kit.pmk.neuroph.parallel.ILearner;

/**
 * Logging utility
 *
 */
public class Log {

	private final static String CSV_HEADER = "name;numThreads;runs;avgEr;confErLo;confErHi;avgTime;confTimeLo;confTimeHi;sumEr;sumTime";
	private final static char DELIM = '~';
	private final static String DEBUG = "[DEBUG]";
	private final static String INFO = "[INFO] ";
	private final static String SCORE = "[SCORE]";
	private static List<String> log = new ArrayList<String>();
	private static List<String> results = new ArrayList<String>();
	private static DateFormat dateFormat = new SimpleDateFormat(
			"yyyy-MM-dd HH:mm:ss");
	private static boolean verbose = false;
	private final static String EOL = System.lineSeparator();

	/**
	 * Generic logging on debug level.
	 * @param tag
	 * @param msg
	 */
	public static void debug(String tag, String msg) {
		log(DEBUG, tag, msg);
	}

	/**
	 * Generic logging on info level.
	 * @param tag
	 * @param msg
	 */
	public static void info(String tag, String msg) {
		log(INFO, tag, msg);
	}

	/**
	 * Log {@link Score} of an experiment.
	 * @param tag
	 * @param score
	 */
	public static void logScore(String tag, Score score) {
		log(SCORE, tag, score.toString());
		results.add(score.toCsv());
	}

	private static void log(String level, String tag, String msg) {
		StringBuilder logSb = new StringBuilder();
		Date date = new Date();
		logSb.append(dateFormat.format(date));
		logSb.append(" ");
		logSb.append(level);
		logSb.append(" ");
		logSb.append(tag);
		logSb.append(" ");
		logSb.append(DELIM);
		logSb.append(" ");
		logSb.append(msg);
		String logString = logSb.toString();
		log.add(logString);
		if (verbose) {
			System.out.println(logString);
		}
	}

	/**
	 * Method writes information of [INFO] level to specified file.
	 * @param filepath
	 * @throws IOException
	 */
	public static void writeAsInfLog(String directory, String filename) throws IOException {
		File dir = new File(directory);
		dir.mkdir();
		String filepath = dir.getPath() + File.separator + "info-" + filename + (new Date()).hashCode() + ".txt";
		writeToFile(log, filepath, "Info Log File", false);
	}
	
	/**
	 * Method writes information of [DEBUG] level and lower levels to specified file.
	 * @param filepath
	 * @throws IOException
	 */
	public static void writeAsDebugLog(String directory, String filename) throws IOException {
		File dir = new File(directory);
		dir.mkdir();
		String filepath = dir.getPath() + File.separator + "debug-" + filename + (new Date()).hashCode() + ".txt";
		writeToFile(log, filepath, "Debug Log File", true);
	}
	
	/**
	 * Method writes scores of experiments to specified csv file.
	 * @param filepath of csv file
	 * @throws IOException
	 */
	public static void writeAsCsvResult(String directory, String filename) throws IOException {
		File dir = new File(directory);
		dir.mkdir();
		String filepath = dir.getPath() + File.separator + "score-" +filename + (new Date()).hashCode() + ".csv";
		writeToFile(results, filepath, CSV_HEADER, false);
	};

	private static void writeToFile(List<String> list, String filepath,
			String title, boolean debug) throws IOException {
		BufferedWriter writer = null;
		try {
			writer = new BufferedWriter(new FileWriter(filepath));
			writer.write(title.toUpperCase() + EOL);
			for (String logstring : list) {
				if (debug) { // debug + info
					writer.write(logstring);
					writer.write(EOL);
				} else if (!logstring.contains(DEBUG)) { // only info level
					writer.write(logstring);
					writer.write(EOL);
				}
			}
		} finally {
			writer.close();
		}
	}

	/**
	 * if given {@link state} is true {@link Log} to standard output stream, else no dynamic logging enabled.
	 * @param state
	 */
	public static void setVerbose(boolean state) {
		verbose = state;
	}

	/**
	 * Resets internal state of Logger.
	 */
	public static void resetLoggingInstance() {
		log = new ArrayList<String>();
		results = new ArrayList<String>();
		verbose = false;
	}
	
	//only for testing
	public static void main(String[] args) {
		resetLoggingInstance();
		setVerbose(true);
		debug(Log.class.getSimpleName(), "Testing debug message");
		info(Log.class.getSimpleName(), "Testing debug message");
		logScore(Log.class.getSimpleName(), new Score(1, new ILearner() {
			
			@Override
			public void resetToUnlearnedState() {				
			}
			
			@Override
			public void learn(DataSet trainingSet) {				
			}
			
			@Override
			public int getNumberOfThreads() {
				return 42;
			}
			
			@Override
			public NeuralNetwork getNeuralNetwork() {
				return null;
			}
			
			@Override
			public String getDescription() {
				return "DummyLearner";
			}
		}));
		try {
			writeAsInfLog(".","info");
			writeAsDebugLog(".", "debug");
			writeAsCsvResult(".", "result");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
