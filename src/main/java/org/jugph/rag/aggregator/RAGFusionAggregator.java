package org.jugph.rag.aggregator;

import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.aggregator.ContentAggregator;
import dev.langchain4j.rag.content.aggregator.ReciprocalRankFuser;
import dev.langchain4j.rag.query.Query;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;


public class RAGFusionAggregator implements ContentAggregator {

    private static final Logger log = LoggerFactory.getLogger(RAGFusionAggregator.class);

    public RAGFusionAggregator() {
    }

    @Override
    public List<Content> aggregate(Map<Query, Collection<List<Content>>> queryToContents) {
        Map<Query, List<Content>> fused = fuse(queryToContents);
//        log.info("fused_before_RRF: {}", fused);
        return ReciprocalRankFuser.fuse(fused.values());
    }

    protected Map<Query, List<Content>> fuse(Map<Query, Collection<List<Content>>> queryToContents) {
        Map<Query, List<Content>> fused = new LinkedHashMap<>();
        for (Query query : queryToContents.keySet()) {
            Collection<List<Content>> contents = queryToContents.get(query);
            fused.put(query, ReciprocalRankFuser.fuse(contents));
        }
//        log.info("fused_after_RRF: {}", fused);
        return fused;
    }
}
