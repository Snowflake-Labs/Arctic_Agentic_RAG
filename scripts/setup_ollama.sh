
# Check if ollama is installed, or install it
if ! command -v ollama &> /dev/null
then
    echo "ollama not found. Please install it."
else
    version=$(ollama --version | sed 's/.* //')
    # if [[ "$version" != "0.4.7" ]]; then
    #     echo "ollama version $version found, but 0.4.7 is required."
    # fi
fi

# setup server side
model="llama3.1:70b"
ollama serve &
ollama pull $model &
ollama run $model &
