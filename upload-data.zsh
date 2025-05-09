
SERVER_NAME="amethyst"


#### Upload blender-gen-dataset things

[ -d "./blender-gen-dataset" ] || exit 1

# Pack config, environment and models
(
cat <<EOF
./blender-gen-dataset/config.toml
./blender-gen-dataset/models
./blender-gen-dataset/environment
EOF
) | tar czf datasets-data.tar.gz --files-from=-

scp datasets-data.tar.gz "$SERVER_NAME":"~/"


