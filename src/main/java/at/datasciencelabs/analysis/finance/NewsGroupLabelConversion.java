package at.datasciencelabs.analysis.finance;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Optional;
import java.util.stream.Stream;

class NewsGroupLabelConversion {

    public double convert(String path) throws IOException {
        URL url = new URL(path);
        String urlPath = url.getPath();
        Path topLevelDirectory = Paths.get(urlPath).getParent().getParent();
        long count = Files.list(topLevelDirectory).count();
        Stream<Path> subDirs =  Files.list(topLevelDirectory);

        HashSet<Long> labels = new HashSet<>();
        subDirs.forEach(p -> labels.add(p.getName(p.getNameCount()-1).hashCode() % count));
        Optional<Long> min = labels.stream().min(Long::compareTo);

        Paths.get(urlPath).getName(Paths.get(urlPath).getNameCount()-3).getFileSystem();
        String categoryPath = Paths.get(urlPath).getName(Paths.get(urlPath).getNameCount()-2).toString();
        return (double)(categoryPath.hashCode()%count - min.orElse(0L));
    }

}
