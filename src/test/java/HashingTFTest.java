import com.google.common.collect.Lists;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.junit.Test;

import java.util.List;

/**
 * Created by phil3k on 01.11.16.
 */
public class HashingTFTest {

    @Test
    public void testHashingTF(){

        List<String> sentences = Lists.newArrayList("PAYLIFE ABRECHNUNG VOM 29.12.2015 EINZUG SIX PAYMENT SERVICES (AUSTRIA) GMBH", "MERKUR 1100 1100P K1 15.06.UM 17.55");


        HashingTF hashingTF = new HashingTF();
        Vector tfVector = hashingTF.transform(sentences);

    }

}
