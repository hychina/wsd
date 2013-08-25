package wsd.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

public class SortUtil {
	
	/**
	 * sort the entries of the map by the value of the entries in descending order
	 * @param map
	 * @return
	 */
	public static List<Map.Entry<String, Double>> sortMapDesc(Map<String, Double> map) {
		ArrayList<Map.Entry<String, Double>> list = new ArrayList<Map.Entry<String, Double>>(map.entrySet());
		Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
			public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
				Double v1 = o1.getValue();
				Double v2 = o2.getValue();
				return -v1.compareTo(v2);
			}
		});
		return list;
	}

}
