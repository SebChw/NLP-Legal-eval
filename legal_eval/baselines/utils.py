def _create_embeddings(examples, embed_model, split_len):
    all_embeddings = []
    all_labels = []

    for example_token, example_tags in zip(examples["tokens"], examples["ner_tags"]):
        for start_pos in range(0, len(example_token), split_len):
            if start_pos + split_len > len(example_token):
                start_pos = max([0, len(example_token) - split_len])

            end_pos = start_pos + split_len
            tokens = example_token[start_pos:end_pos]
            labels = example_tags[start_pos:end_pos]
            embeddings = [embed_model.get_word_vector(word) for word in tokens]

            all_embeddings.append(embeddings)
            all_labels.append(labels)

    return {
        "embeddings": all_embeddings,
        "label": all_labels,
    }  # input_ids to fool Trainer


def prepare_dataset_staticemb(dataset, embed_model, split_len=64):
    # We need custom data collator
    # We don't need custom tokenizer it's not necessary
    dataset = dataset.map(
        _create_embeddings,
        batched=True,
        remove_columns=["tokens", "ner_tags"],
        fn_kwargs={"embed_model": embed_model, "split_len": split_len},
        load_from_cache_file=False,
    )
    dataset.set_format("pt", columns=["embeddings", "label"])
    return dataset
