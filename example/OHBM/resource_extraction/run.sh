set -x  # turn on verbose/debug mode

# your commands here
echo "Running resource extraction..."

structsense-cli extract \
--source paper_1909.11229v2.pdf \
--config config.yaml \
--env_file .env_ohbm_hackathon
--save_file result.json
