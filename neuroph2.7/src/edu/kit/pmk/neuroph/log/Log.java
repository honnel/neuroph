package edu.kit.pmk.neuroph.log;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class Log {

	private final static char DELIM = '~';
	private final static char DEBUG = '§';
	private static List<String> log = new ArrayList<String>();
	private static DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss");;

	public static void debug(String tag, String msg) {
		log.add(DEBUG + generateLogString(tag, msg));
	}

	public static void info(String tag, String msg) {
		log.add(generateLogString(tag, msg));
	}
	
	private static String generateLogString(String tag, String msg) {
		Date date = new Date();	
		return dateFormat.format(date) + tag + DELIM + msg;
	}
	
	public static void writeAsCSVFile(String filepath) throws IOException {
		writeToFile(filepath, "Date;AvgTime;loTime;hiTime;AvgError;loError;hiError", false);
	}
	
	public static void writeAsDebugInfo(String filepath) throws IOException {
		writeToFile(filepath, "", true);
	}

	private static void writeToFile(String filepath, String title, boolean debug)
			throws IOException {
		BufferedWriter writer = null;		
		try {
			writer = new BufferedWriter(new FileWriter(filepath));
			writer.write(title.toUpperCase() + "\n");
			for (String logstring : log) {
				if (debug) {
					writer.write(logstring);
				} else if (logstring.charAt(0) != DEBUG) {
					writer.write(logstring);
				}
			}
		} finally {
			writer.close();
		}
	}
	
	public static void resetLoggingInstance() {
		log = new ArrayList<String>(); 
	}

}
