package at.datasciencelabs.analysis.finance;

import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Created by phil3k on 19.11.16.
 */
public class NewsGroupLabelConversionTest {

    @Test
    public void testLabelGeneration() throws IOException {
        NewsGroupLabelConversion newsGroupLabelConversion = new NewsGroupLabelConversion();
        System.out.println(newsGroupLabelConversion.convert("file:/home/phil3k/projects/spark-playground/spark-playground/20_newsgroup/alt.atheism/51119"));
    }

    @Test
    public void testLabelsAreDistinct() throws IOException {
        long count = Files.list(Paths.get("/home/phil3k/projects/spark-playground/spark-playground/20_newsgroup/alt.atheism/51119").getParent().getParent()).count();
        Stream<Path> subDirs =  Files.list(Paths.get("/home/phil3k/projects/spark-playground/spark-playground/20_newsgroup/alt.atheism/51119").getParent().getParent());

        HashSet<Long> labels = new HashSet<>();
        subDirs.forEach(p -> labels.add(p.getName(p.getNameCount()-1).hashCode() % count));
        Optional<Long> min = labels.stream().min(Long::compareTo);
        Set<Long> finalLabels = labels.stream().map(label -> label-min.orElse(0L)).collect(Collectors.toSet());
        finalLabels.forEach(System.out::println);

    }

}