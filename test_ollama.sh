#!/bin/bash

# Script to test Ollama API endpoints
# Assumes Ollama server is running at http://localhost:11434
# Uses jq for pretty-printing JSON if available.

BASE_URL="http://localhost:11434/api"
MODEL_TO_PULL="phi3:mini" # A relatively small model for testing
PULLED_MODEL_NAME=$MODEL_TO_PULL
COPIED_MODEL_NAME="${MODEL_TO_PULL}-copy"
# Replace : for valid model name, and ensure it's a valid format
CUSTOM_MODEL_NAME="my-custom-model-from-$(echo "$MODEL_TO_PULL" | sed 's/:/_/g')"
DUMMY_FILE="dummy_model_content.bin"
DUMMY_FILE_DIGEST=""

# Function to make requests and print output
# $1: HTTP Method (GET, POST, DELETE, HEAD)
# $2: Endpoint path (e.g., /tags)
# $3: Optional JSON data for POST/PUT/DELETE or file path for POST_FILE
# $4: Optional description
make_request() {
    local method="$1"
    local endpoint="$2"
    local data_or_file="$3"
    local description="$4"
    local full_url="${BASE_URL}${endpoint}"
    # -s for silent, -w to output status, -L to follow redirects
    # Removed %{json done_reason} as it's not standard and caused errors.
    # Removed trailing \\n from -w format string.
    local curl_opts=("-s" "-L" "-w" "\\nHTTP_STATUS: %{http_code}")

    echo "----------------------------------------------------------------------"
    if [ -n "$description" ]; then
        echo "Testing: $description ($method $full_url)"
    else
        echo "Testing: $method $full_url"
    fi

    if [ "$method" == "POST_FILE" ]; then
        local file_path="$data_or_file"
        # For POST_FILE, $endpoint is the full URL including digest
        echo "Uploading file: $file_path to $endpoint" # $endpoint is already full_url here
        response=$(curl "${curl_opts[@]}" -X POST -T "$file_path" "$endpoint")
    elif [ -n "$data_or_file" ] && [ "$method" != "GET" ] && [ "$method" != "HEAD" ]; then
        echo "Request Data:"
        echo "$data_or_file" # Print data for verification
        response=$(curl "${curl_opts[@]}" -X "$method" -H "Content-Type: application/json" -d "$data_or_file" "$full_url")
    else
        response=$(curl "${curl_opts[@]}" -X "$method" "$full_url")
    fi

    echo "Response:"
    # Response from curl with -w "\\nHTTP_STATUS: %{http_code}" will be:
    # <body_content_possibly_multiline>
    # HTTP_STATUS: <http_code>
    # So, status is the last line, body is everything before it.

    response_status_line=$(echo "$response" | tail -n 1)

    if [[ "$response_status_line" == HTTP_STATUS:* ]]; then
        response_body=$(echo "$response" | sed '$d') # Remove last line (status line)
        response_status="$response_status_line"
    else
        # If the last line isn't the status, assume the whole response is body (e.g. curl error before -w took effect)
        response_body="$response"
        response_status="HTTP_STATUS: (unknown / not found in response)"
    fi
    
    # Removed parsing for response_done_reason as it's no longer in curl_opts

    if command -v jq &> /dev/null && echo "$response_body" | jq . &> /dev/null; then
        echo "$response_body" | jq .
    else
        echo "$response_body"
    fi
    echo "$response_status"
    # Removed printing of response_done_reason
    echo "----------------------------------------------------------------------"
    echo ""
}

# --- Start Tests ---

echo "Ollama API Test Script"
echo "Using model: $MODEL_TO_PULL for tests requiring a model."
echo "Ensure Ollama server is running at $BASE_URL"
echo "Ensure jq is installed for pretty JSON output."
echo ""

# 0. Check Ollama server status (/)
echo "----------------------------------------------------------------------"
echo "Testing: Ollama Server Status (GET http://localhost:11434/)"
curl -s http://localhost:11434/
echo ""
echo "----------------------------------------------------------------------"
echo ""

# 1. List Local Models (/api/tags) - Initial state
make_request "GET" "/tags" "" "List Local Models (Initial)"

# 2. Pull a Model (/api/pull)
# This is crucial for many subsequent tests
PULL_DATA_NON_STREAMING=$(cat <<EOF
{
  "name": "$MODEL_TO_PULL",
  "stream": false,
  "insecure": false
}
EOF
)
make_request "POST" "/pull" "$PULL_DATA_NON_STREAMING" "Pull Model ($MODEL_TO_PULL, non-streaming)"

PULL_DATA_STREAMING=$(cat <<EOF
{
  "name": "$MODEL_TO_PULL",
  "stream": true,
  "insecure": false
}
EOF
)
# For streaming, output will be multiple JSON objects.
echo "----------------------------------------------------------------------"
echo "Testing: Pull Model ($MODEL_TO_PULL, streaming) (POST ${BASE_URL}/pull)"
echo "Request Data:"
echo "$PULL_DATA_STREAMING"
echo "Response (raw stream):"
curl -s -N -X POST -H "Content-Type: application/json" -d "$PULL_DATA_STREAMING" "${BASE_URL}/pull"
echo "" # Add a newline after stream
echo "HTTP_STATUS: (Streamed, check output above for success/failure messages)"
echo "----------------------------------------------------------------------"
echo ""

# 3. List Local Models (/api/tags) - After pull
make_request "GET" "/tags" "" "List Local Models (After pulling $MODEL_TO_PULL)"

# 4. Show Model Information (/api/show)
SHOW_DATA=$(cat <<EOF
{
  "name": "$PULLED_MODEL_NAME"
}
EOF
)
make_request "POST" "/show" "$SHOW_DATA" "Show Model Information ($PULLED_MODEL_NAME)"

SHOW_DATA_VERBOSE=$(cat <<EOF
{
  "name": "$PULLED_MODEL_NAME",
  "verbose": true
}
EOF
)
make_request "POST" "/show" "$SHOW_DATA_VERBOSE" "Show Model Information ($PULLED_MODEL_NAME, verbose)"


# 5. Generate Completion (/api/generate)
GENERATE_DATA_NON_STREAMING=$(cat <<EOF
{
  "model": "$PULLED_MODEL_NAME",
  "prompt": "Why is the sky blue?",
  "stream": false
}
EOF
)
make_request "POST" "/generate" "$GENERATE_DATA_NON_STREAMING" "Generate Completion (Non-streaming)"

GENERATE_DATA_STREAMING=$(cat <<EOF
{
  "model": "$PULLED_MODEL_NAME",
  "prompt": "Tell me a short story about a robot.",
  "stream": true
}
EOF
)
echo "----------------------------------------------------------------------"
echo "Testing: Generate Completion (Streaming) (POST ${BASE_URL}/generate)"
echo "Request Data:"
echo "$GENERATE_DATA_STREAMING"
echo "Response (raw stream):"
curl -s -N -X POST -H "Content-Type: application/json" -d "$GENERATE_DATA_STREAMING" "${BASE_URL}/generate"
echo "" # Add a newline after stream
echo "HTTP_STATUS: (Streamed, check output above for success/failure messages)"
echo "----------------------------------------------------------------------"
echo ""

# 6. Generate Chat Completion (/api/chat)
CHAT_DATA_NON_STREAMING=$(cat <<EOF
{
  "model": "$PULLED_MODEL_NAME",
  "messages": [
    {"role": "user", "content": "Hello, who are you?"}
  ],
  "stream": false
}
EOF
)
make_request "POST" "/chat" "$CHAT_DATA_NON_STREAMING" "Generate Chat Completion (Non-streaming)"

CHAT_DATA_STREAMING=$(cat <<EOF
{
  "model": "$PULLED_MODEL_NAME",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "stream": true
}
EOF
)
echo "----------------------------------------------------------------------"
echo "Testing: Generate Chat Completion (Streaming) (POST ${BASE_URL}/chat)"
echo "Request Data:"
echo "$CHAT_DATA_STREAMING"
echo "Response (raw stream):"
curl -s -N -X POST -H "Content-Type: application/json" -d "$CHAT_DATA_STREAMING" "${BASE_URL}/chat"
echo "" # Add a newline after stream
echo "HTTP_STATUS: (Streamed, check output above for success/failure messages)"
echo "----------------------------------------------------------------------"
echo ""

# 7. Generate Embeddings (/api/embed)
# Note: /api/embeddings is deprecated, using /api/embed
EMBED_DATA=$(cat <<EOF
{
  "model": "$PULLED_MODEL_NAME",
  "input": "This is a test sentence for embeddings."
}
EOF
)
make_request "POST" "/embed" "$EMBED_DATA" "Generate Embeddings"

# 8. Copy a Model (/api/copy)
COPY_DATA=$(cat <<EOF
{
  "source": "$PULLED_MODEL_NAME",
  "destination": "$COPIED_MODEL_NAME"
}
EOF
)
make_request "POST" "/copy" "$COPY_DATA" "Copy Model ($PULLED_MODEL_NAME to $COPIED_MODEL_NAME)"

# Verify copy by listing models
make_request "GET" "/tags" "" "List Local Models (After copying to $COPIED_MODEL_NAME)"

# 9. List Running Models (/api/ps)
make_request "GET" "/ps" "" "List Currently Running/Loaded Models"

# 10. Create a Model (/api/create)
# Requires a Modelfile content.
# The Modelfile FROM instruction should use a model that exists.
CREATE_MODEL_DATA=$(cat <<EOF
{
  "name": "$CUSTOM_MODEL_NAME",
  "modelfile": "FROM $PULLED_MODEL_NAME\\nSYSTEM You are a custom test assistant.\\nPARAMETER temperature 0.7",
  "stream": false
}
EOF
)
make_request "POST" "/create" "$CREATE_MODEL_DATA" "Create Model ($CUSTOM_MODEL_NAME from $PULLED_MODEL_NAME, non-streaming)"

CREATE_MODEL_DATA_STREAMING=$(cat <<EOF
{
  "name": "${CUSTOM_MODEL_NAME}-streaming",
  "modelfile": "FROM $PULLED_MODEL_NAME\\nSYSTEM You are another custom test assistant.\\nPARAMETER temperature 0.8",
  "stream": true
}
EOF
)
echo "----------------------------------------------------------------------"
echo "Testing: Create Model (${CUSTOM_MODEL_NAME}-streaming from $PULLED_MODEL_NAME, streaming) (POST ${BASE_URL}/create)"
echo "Request Data:"
echo "$CREATE_MODEL_DATA_STREAMING"
echo "Response (raw stream):"
curl -s -N -X POST -H "Content-Type: application/json" -d "$CREATE_MODEL_DATA_STREAMING" "${BASE_URL}/create"
echo "" # Add a newline after stream
echo "HTTP_STATUS: (Streamed, check output above for success/failure messages)"
echo "----------------------------------------------------------------------"
echo ""


# Verify model creation
make_request "GET" "/tags" "" "List Local Models (After creating custom models)"

# 11. Blobs
# 11a. Create a dummy file for blob testing
echo "This is dummy content for blob testing." > "$DUMMY_FILE"
if [[ "$(uname)" == "Darwin" ]]; then # macOS
    DUMMY_FILE_DIGEST=$(shasum -a 256 "$DUMMY_FILE" | awk '{print $1}')
else # Linux
    DUMMY_FILE_DIGEST=$(sha256sum "$DUMMY_FILE" | awk '{print $1}')
fi
echo "Dummy file '$DUMMY_FILE' created with SHA256 digest: $DUMMY_FILE_DIGEST"

# 11b. Check Blob Existence (HEAD /api/blobs/sha256:<digest>) - Expected 404 initially
make_request "HEAD" "/blobs/sha256:$DUMMY_FILE_DIGEST" "" "Check Blob Existence (Non-existent blob)"

# 11c. Create a Blob (POST /api/blobs/sha256:<digest>)
# The endpoint for POST_FILE is the full URL
make_request "POST_FILE" "${BASE_URL}/blobs/sha256:$DUMMY_FILE_DIGEST" "$DUMMY_FILE" "Create Blob from $DUMMY_FILE"

# 11d. Check Blob Existence (HEAD /api/blobs/sha256:<digest>) - Expected 200 after creation
make_request "HEAD" "/blobs/sha256:$DUMMY_FILE_DIGEST" "" "Check Blob Existence (After creating blob)"

# 12. Push a Model (/api/push)
# Ollama push typically requires a configured remote registry (e.g., ollama.ai or private).
# This test will likely fail if not configured.
# Replace 'yournamespace' with your actual namespace if you intend to test this for real.
# For safety, this is a placeholder and will likely result in an error unless the user has
# a model named "yournamespace/phi3:mini" or similar locally and is logged in.
PUSH_MODEL_NAME_FORMATTED="yournamespace/$PULLED_MODEL_NAME" # Assuming PULLED_MODEL_NAME is like 'model:tag'
# If PULLED_MODEL_NAME is just 'model', then this is fine. If it's 'model:tag', Ollama push usually wants 'namespace/model:tag'
# However, the API docs for push just say "name" of the model. Let's assume PULLED_MODEL_NAME is what's needed.
# For robust testing against a registry, PUSH_MODEL_NAME_FORMATTED might need to be "yournamespace/$(echo $PULLED_MODEL_NAME | cut -d':' -f1):$(echo $PULLED_MODEL_NAME | cut -d':' -f2)"
# But for now, let's use a simpler form, assuming the user might have a model like "yournamespace/phi3:mini" if they test this.
# The original script used PUSH_MODEL_NAME_FORMATTED="yournamespace/$PULLED_MODEL_NAME" which is fine if PULLED_MODEL_NAME is just the model name without the tag.
# Let's stick to the original intention for PUSH_MODEL_NAME_FORMATTED for now.
PUSH_DATA_NON_STREAMING=$(cat <<EOF
{
  "name": "$PUSH_MODEL_NAME_FORMATTED",
  "insecure": true,
  "stream": false
}
EOF
)
echo "----------------------------------------------------------------------"
echo "Attempting to Push Model (POST ${BASE_URL}/push, non-streaming)"
echo "Model: $PUSH_MODEL_NAME_FORMATTED"
echo "NOTE: This will likely fail if you haven't configured a remote registry"
echo "      and authenticated. You might need to log in via 'ollama login'."
echo "      Replace 'yournamespace' with your actual namespace if testing for real."
echo "Request Data:"
echo "$PUSH_DATA_NON_STREAMING"
response=$(curl -s -L -w "\\nHTTP_STATUS: %{http_code}" -X POST -H "Content-Type: application/json" -d "$PUSH_DATA_NON_STREAMING" "${BASE_URL}/push")

response_status_line=$(echo "$response" | tail -n 1)
if [[ "$response_status_line" == HTTP_STATUS:* ]]; then
    response_body=$(echo "$response" | sed '$d')
    response_status="$response_status_line"
else
    response_body="$response"
    response_status="HTTP_STATUS: (unknown / not found in response)"
fi

if command -v jq &> /dev/null && echo "$response_body" | jq . &> /dev/null; then
    echo "$response_body" | jq .
else
    echo "$response_body"
fi
echo "$response_status"
echo "----------------------------------------------------------------------"
echo ""

PUSH_DATA_STREAMING=$(cat <<EOF
{
  "name": "$PUSH_MODEL_NAME_FORMATTED",
  "insecure": true,
  "stream": true
}
EOF
)
echo "----------------------------------------------------------------------"
echo "Attempting to Push Model (POST ${BASE_URL}/push, streaming)"
echo "Model: $PUSH_MODEL_NAME_FORMATTED"
echo "(See notes above about push configuration)"
echo "Request Data:"
echo "$PUSH_DATA_STREAMING"
echo "Response (raw stream):"
curl -s -N -L -X POST -H "Content-Type: application/json" -d "$PUSH_DATA_STREAMING" "${BASE_URL}/push"
echo ""
echo "HTTP_STATUS: (Streamed, check output above for success/failure messages)"
echo "----------------------------------------------------------------------"
echo ""


# --- Cleanup ---
echo "Starting Cleanup..."

# 13. Delete Models (/api/delete)
DELETE_COPIED_MODEL_DATA=$(cat <<EOF
{ "name": "$COPIED_MODEL_NAME" }
EOF
)
make_request "DELETE" "/delete" "$DELETE_COPIED_MODEL_DATA" "Delete Copied Model ($COPIED_MODEL_NAME)"

DELETE_CUSTOM_MODEL_DATA=$(cat <<EOF
{ "name": "$CUSTOM_MODEL_NAME" }
EOF
)
make_request "DELETE" "/delete" "$DELETE_CUSTOM_MODEL_DATA" "Delete Custom Model ($CUSTOM_MODEL_NAME)"

DELETE_CUSTOM_STREAMING_MODEL_DATA=$(cat <<EOF
{ "name": "${CUSTOM_MODEL_NAME}-streaming" }
EOF
)
make_request "DELETE" "/delete" "$DELETE_CUSTOM_STREAMING_MODEL_DATA" "Delete Custom Streaming Model (${CUSTOM_MODEL_NAME}-streaming)"


# Optionally, delete the initially pulled model if you want a clean state
# UNCOMMENT THE FOLLOWING LINES TO DELETE THE BASE PULLED MODEL
# echo "Attempting to delete the base pulled model: $PULLED_MODEL_NAME"
# DELETE_PULLED_MODEL_DATA=\$(cat <<EOF
# { "name": "$PULLED_MODEL_NAME" }
# EOF
# )
# make_request "DELETE" "/delete" "\$DELETE_PULLED_MODEL_DATA" "Delete Pulled Model (\$PULLED_MODEL_NAME)"


# List models after deletions
make_request "GET" "/tags" "" "List Local Models (After Deletions)"

# Remove dummy file
if [ -f "$DUMMY_FILE" ]; then
    rm "$DUMMY_FILE"
    echo "Removed dummy file: $DUMMY_FILE"
fi

echo ""
echo "Ollama API Test Script Finished."
echo "Review the output above for results of each test."
echo "Remember to check streaming outputs directly as they are not fully parsed by this script."

# --- End of Script ---
