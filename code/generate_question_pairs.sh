#!/bin/bash

OUTPUT_DIR="questions/"
GENERATE_QUESTIONS_ONLY="True"
READ_GRAPH="True"

# Loop through all the JSON files in the topic folder
for TOPIC_FILE in ../data/topic/*.json; do
    # Extract the domain and topic name from the file name
    FILE_NAME=$(basename "$TOPIC_FILE")
    DOMAIN_NAME=$(echo "$FILE_NAME" | cut -d'_' -f2)
    TOPIC_NAME=$(echo "$FILE_NAME" | cut -d'_' -f3 | cut -d'.' -f1)

    GRAPH_PATH="graph/${TOPIC_NAME}_graph.gpickle"

    echo "Domain: $DOMAIN_NAME, Topic: $TOPIC_NAME"
    python data_prep.py -o="$OUTPUT_DIR" --topics="$TOPIC_FILE" --generate_questions_only="$GENERATE_QUESTIONS_ONLY" --read_graph="$READ_GRAPH" --graph_path="$GRAPH_PATH"
done

# python data_prep.py -o=questions/ --topics=topic/topic_places_landmark.json --generate_questions_only=True --read_graph=True --graph_path=graph/landmark_graph.gpickle