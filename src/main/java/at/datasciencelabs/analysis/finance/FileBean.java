package at.datasciencelabs.analysis.finance;

import java.io.Serializable;

/**
 * Created by phil3k on 08.11.16.
 */
public class FileBean implements Serializable {

    private String key;
    private String value;

    public FileBean() {

    }

    FileBean(String s, String s1) {
        this.key = s;
        this.value = s1;
    }

    public String getKey() {
        return key;
    }

    public void setKey(String key) {
        this.key = key;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return "FileBean{" +
                "key='" + key + '\'' +
                ", value='" + value + '\'' +
                '}';
    }
}
