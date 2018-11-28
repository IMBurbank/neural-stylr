#!/bin/sh

##### CONSTANTS #####
DEFAULT_TAG='neural-stylr'
DEFAULT_FILE='model/dockerfiles/dev.Dockerfile'
DEFAULT_CONTEXT='model/'


##### VARIABLES #####
build_context=


##### FLAGS #####
build_opts=			# flag -b
tag_suffix='-gpu'   # flag -c (use-cpu)
file_path=""		# flag -f
tag=""				# flag -t

while getopts ":h:b:f:c:t:" flag; do
	case "${flag}" in
		h) 
			echo "HELP STUB"
			exit 0
			;;
		b) 
			build_opts="${OPTARG} ${build_opts}"
			;;
		f) 
			file_path="${OPTARG}"
			;;
		c) 
			tag_suffix=
			;;
		t) 
			tag="${OPTARG}"
			;;
		*) 
			echo "ERROR: Invalid flag ${flag}"
			exit 1
			;;
	esac
done
shift `expr $OPTIND - 1`


##### MAIN #####
build_context=$1

[ -z "${file_path}" ] && file_path=${DEFAULT_FILE}
[ -z "${tag}" ] && tag="${DEFAULT_TAG}${tag_suffix}"
[ -z "${build_context}" ] && build_context=${DEFAULT_CONTEXT}
[ ! -z "${tag_suffix}" ] && build_opts="${build_opts} --build-arg USE_GPU=True"

cat << ECHO
---------------------------------------
Beginning Docker Build
Dockerfile: ${file_path}
Context:    ${build_context}
Tag:        ${tag}
Build-opts: ${build_opts}
---------------------------------------
ECHO

docker build ${build_opts} -f ${file_path} -t ${tag} ${build_context}