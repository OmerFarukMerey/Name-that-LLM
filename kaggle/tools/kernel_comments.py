"""Print the comment threads on our public competition notebooks (read-only). Run via kglrun.py.

    python kglrun.py python kernel_comments.py [user/kernel-slug ...]

Without arguments it reads every public notebook of ours in the competition. On a notebook each comment
thread is a topic: its first message is the topic's content (only get_topic returns it), the replies are
its comments.
"""
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.discussions.types.discussions_api_service import (ApiGetTopicRequest, ApiListCommentsRequest,
                                                                 ApiListKernelTopicsRequest)
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiListKernelsRequest


def call(method, request):
    time.sleep(1)   # ~60 calls in a burst draw 429 Too Many Requests
    return method(request)


def show(c, depth):
    pad = "    " * depth
    print(f"{pad}- {c.author_name} ({c.author_url}) {str(c.post_date)[:16]} UTC, {c.votes} votes, id {c.id}")
    for line in (c.content or "").splitlines():
        print(f"{pad}  | {line}")
    for r in getattr(c, "replies", None) or []:   # a topic has no replies field: its comments are listed apart
        show(r, depth + 1)


api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as kaggle:
    kernels, discussions = kaggle.kernels.kernels_api_client, kaggle.discussions.discussion_api_client
    refs = sys.argv[1:]
    if not refs:
        r = ApiListKernelsRequest(); r.competition = "name-that-llm"; r.user = "omerfarukmerey"; r.page_size = 100
        res = kernels.list_kernels(r)
        for k in (res.kernels if hasattr(res, "kernels") else res):
            # the listing reports is_private False for private kernels too (2026-11-05), get_kernel does not
            g = ApiGetKernelRequest(); g.user_name, g.kernel_slug = k.ref.split("/")
            if call(kernels.get_kernel, g).metadata.is_private is False:
                refs.append(k.ref)
    for ref in refs:
        r = ApiListKernelTopicsRequest(); r.owner_slug, r.kernel_slug = ref.split("/"); r.page_size = 50
        topics = call(discussions.list_kernel_topics, r).topics or []
        print(f"{ref}: {len(topics)} thread(s)")
        for t in topics:
            g = ApiGetTopicRequest(); g.id = t.id
            show(call(discussions.get_topic, g).topic, 1)
            c = ApiListCommentsRequest(); c.topic_id = t.id; c.page_size = 100
            for cm in call(discussions.list_comments, c).comments or []:
                show(cm, 2)
