<template>
  <div id="app">
    <div class="main">
      <div class="column">
        <div class="sidebar" :class="{show: isMenuShow}">
          <nav id="nav" class="nav">
            <div class="sm-text close-button" @click="isMenuShow=!isMenuShow"><i class="fa fa-close"></i> Close</div>
            <ul class="nav-list">
              <li class="nav-list-item title">
                <a href="#" class="nav-link no-hover">
                  Face<br/><span class="gradient">ReLight</span>
                </a></li>
              <li class="nav-list-item"><a href="https://3duniversum.com" target="_blank" class="nav-link hovered company-link">by 3duniversum</a></li>
              <li class="nav-list-item divider">-----</li>
              <li class="nav-list-item">
                <label class="button">Try your image
                  <input type="file" class="hidden_input" ref="file" @change="handleFileUpload()" :disabled="isBusy" accept="image/png, image/jpeg, image/jpg"/>
                </label>
              </li>
              <li class="nav-list-item divider">-----</li>
              <li class="nav-list-item">
                <div>
                  <h3>Examples</h3>
                  <div class="gallery">
                    <div class="gallery-item" v-for="(item, exampleIndex) in examples" :key="exampleIndex+item" @click="selectExample(item)">
                      <div class="gallery-item-image"
                          :style="{'background-image': `url(/media/output/${item}/ori.jpg)`}">
                      </div>
                    </div>
                  </div>
                </div>
              </li>
            </ul>
          </nav>
        </div>
        <div class="content">
          <button class="button sm-text menu" @click="isMenuShow=!isMenuShow"><i class="fa fa-bars"></i> Menu</button>
          <div class="sm-text title">Face <span class="gradient">ReLight</span></div>
          <section id="examples" class="examples-section">
            <div id="example-images">
              <div class="image-section with-face-relight-output">
                <div class="image-wrap">

                  <img class="target-image cover" :src="imgURL" :class="{'hide': imgIndex!=selectedValue}" @click="selectOrbit('none')" v-for="(imgURL, imgIndex) in imageList" :key="imgIndex+imgURL">
                  <img class="target-image" :src="getOriginalImage">
                  
                  <div class="sh-preview" v-if="isShowingSHPreview">
                    <div class="message">This is just a preview, click "apply" to implement the changes on all sequences</div>
                    <img class="target-image cover" :src="getPreviewSH">
                  </div>

                  <div class="image-wrap-cover" :class="{'show': isBusy}" v-if="isBusy" >{{isBusy}}</div>
                </div>
                <div class="face-relight-output" style="display: block;">
                  <table>
                    <tr>
                      <td>
                        <h3 class="function-title">Light Orbit</h3>
                      </td>
                      <td class="get-pallete">
                        <div class="control-wrapper">
                          <button class="button" @click="selectOrbit('horizontal')" :class="{selected:selectedOrbit == 'horizontal'}">
                            <span class="sm-text">HRZ</span>
                            <span class="m-text">Horizontal</span>
                          </button>
                           <!-- | 
                          <button class="button" @click="selectOrbit('over')" :class="{selected:selectedOrbit == 'over'}">
                            Over
                          </button> -->
                           | 
                          <button class="button" @click="selectOrbit('around')" :class="{selected:selectedOrbit == 'around'}">
                            Around
                          </button>
                        </div>
                      </td>
                    </tr>
                    <tr :style="{'opacity': selectedOrbit==='none' ? 0.2: 1.0}">
                      <td>
                        <h3 class="function-title">Direction</h3>
                      </td>
                      <td class="get-pallete">
                        <div class="control-wrapper">
                          <input type="range" class="control-slider" :disabled="isBusy"
                            :min="minRange" :max="maxRange" step="1" v-model="selectedValue" @focus="resetSH()"
                          >
                        </div>
                      </td>
                    </tr>
                     <tr :style="{'opacity': selectedOrbit==='none' ? 0.2: 1.0}">
                      <td>
                        <h3 class="function-title" v-if="isShowingAdvanceOption">Intensity</h3>
                      </td>
                      <td class="get-pallete">
                        <div class="control-wrapper">
                          <div class="extra-option-button default-hidden" :class="{'show': !isShowingAdvanceOption}"
                            @click="checkForSHPreview">
                            show extra options
                          </div>
                            
                          <input type="range" class="control-slider default-hidden" :class="{'show': isShowingAdvanceOption}" 
                            :disabled="isBusy"
                            :min="minRangeSH" :max="maxRangeSH" step="0.1" v-model="shMul" 
                          >
                          <button class="button special default-hidden" :class="{'show': isShowingAdvanceOption}" 
                          @click="handleUpdateSHMul()" :disabled="!isShowingSHPreview"> Apply </button>
                          
                        </div>
                      </td>
                    </tr>
                  </table>                            
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
  const MEDIA_ROOT = '/media/output'
  import axios from 'axios'
  import Swal from 'sweetalert2'
  import Vue from 'vue'
  
  const range = {
    'horizontal': {
      minRange: 0,
      maxRange: 52,
      midRange: 25
    },
    'over': {
      minRange: 0,
      maxRange: 35,
      midRange: 16
    },
    'around': {
      minRange: 0,
      maxRange: 69,
      midRange: 32
    },
    'none': {
      minRange:0,
      maxRange:0,
      midRange:0
    }
  }

  const defaultOrbit = 'none';
  export default {
    data() {
      return {
        imageList: [],
        minRangeSH: 0.5,
        maxRangeSH: 1.0,
        examples: ['sample_grlee', 'sample_paris','sample_AJ', 'sample_a1', 'sample_aa', 'sample_Faycey', 'sample_kat', 'sample_mal', 'sample_maja'],
        selectedModel: 'sample_AJ',
        selectedOrbit: defaultOrbit,
        selectedValue: range[defaultOrbit].midRange,
        minRange: range[defaultOrbit].minRange,
        maxRange: range[defaultOrbit].maxRange,
        isMenuShow: false,
        isShowingSHPreview: false,
        isBusy: false,
        selectedSHMul: 0.7,
        shMul: 0.7,
        isShowingAdvanceOption: false,
      }
    },
    components: {
    },
    methods: {
      selectOrbit(type) {
        this.selectedOrbit = type
        this.selectedValue = range[type].midRange
        this.minRange = range[type].minRange
        this.maxRange = range[type].maxRange
        this.isShowingSHPreview = false
        this.isShowingAdvanceOption = false
        this.updateImageList();
      },
      selectExample(id) {
        this.selectedModel = id
        this.selectOrbit('none')
        this.isMenuShow = false
      },
      resetSH() {
        this.isShowingSHPreview = false
        this.isShowingAdvanceOption = false

        this.isShowingSHPreview=false
        this.shMul = this.selectedSHMul
      },
      checkForSHPreview() {
        if (!this.isShowingSHPreview && this.selectedOrbit!=='none') {
          this.isShowingSHPreview = true
          this.handleSHPreviewGeneration()
        }
      },
      handleFileUpload() {
        const file = this.$refs.file.files[0];
        if (!!file && !this.isBusy) {
          this.isBusy = true;
          let formData = new FormData();
          formData.append('file', file);
          axios.post( '/',
            formData,
            {
              headers: {
                  'Content-Type': 'multipart/form-data'
              },
              responseType: 'json'
            }
          ).then((results) => {
            if (!!results.data.is_face_found) {
              this.selectExample(results.data.upload_id);
              this.isBusy = false;
            } else {
              Swal.fire({
                icon: 'error',
                title: 'We could not detect a face in this picture',
                text: 'Please upload an image with atleast on visible face',
              });
              this.isBusy = false;
            }
          })
          .catch((error) => {
            console.log('err', error)
            Swal.fire({
                icon: 'error',
                title: 'Oops...',
                text: 'Server is having a difficulty processing your file, please try again in a few minutes...',
              });

            this.isBusy = false;
          });
        }
      },
      generateImageURL(val) {
        const defaultURL = `${MEDIA_ROOT}/${this.selectedModel}/ori.jpg`;
        let id = this.selectedOrbit==='over' ? this.maxRange - val : val;
        if (this.selectedOrbit === 'around') {
          id = this.maxRange - val - 16;
          if (id < 0) {
            id += this.maxRange
          }
        }
                
        return this.selectedOrbit==='none' ? defaultURL : `${MEDIA_ROOT}/${this.selectedModel}/${this.selectedOrbit}_${id}_${(+this.selectedSHMul).toFixed(2)}.jpg`
      },
      handleUpdateSHMul() {               
        if (!this.isBusy) {
          this.isBusy = true;
          this.selectedSHMul = this.shMul
        
          axios.post( '/create-sh-presets',{
              file_id: this.selectedModel,
              sh_mul: +this.selectedSHMul
            }
          ).then((results) => {
            // this.selectedModel = results.data.upload_id;
            this.updateImageList()
            setTimeout(()=> {
              this.isBusy = false;
              this.isShowingAdvanceOption = false
              this.isShowingSHPreview = false
            }, 500)
          })
          .catch((error) => {
            console.log('err', error)
            Swal.fire({
                icon: 'error',
                title: 'Oops...',
                text: 'Server is having a difficulty processing your file, please try again in a few minutes...',
              });

            this.isBusy = false;
          });
        }
      },
      handleSHPreviewGeneration() {
        if (!this.isBusy) {
          this.isBusy = true;
        
          let id = this.selectedOrbit==='over' ? this.maxRange - this.selectedValue : this.selectedValue;
          if (this.selectedOrbit === 'around') {
            id = this.maxRange - this.selectedValue - 16;
            if (id < 0) {
              id += this.maxRange
            }
          }
          
          axios.post( '/create-sh-previews',{
              file_id: this.selectedModel,
              sh_id: +id,
              preset_name: this.selectedOrbit,
            }
          ).then((results) => {
            // this.selectedModel = results.data.upload_id;   
            this.updateImageList();         
            setTimeout(()=> {
              this.isBusy = false;
              this.isShowingAdvanceOption = true
            }, 500)
          })
          .catch((error) => {
            console.log('err', error)
            Swal.fire({
                icon: 'error',
                title: 'Oops...',
                text: 'Server is having a difficulty processing your request, please try again...',
              });

            this.isBusy = false;
          });
        }
      },
      updateImageList() {        
        const list = new Array(this.maxRange + 1)

        for (let val=this.minRange ; val <= this.maxRange; val++) {
          list[val] = this.generateImageURL(val)
        }

        this.imageList = list
      },
    },
    computed: { 
      getOriginalImage() {
        return `${MEDIA_ROOT}/${this.selectedModel}/ori.jpg`
      },
      getPreviewSH() {
        let id = this.selectedOrbit==='over' ? this.maxRange - this.selectedValue : this.selectedValue;
        if (this.selectedOrbit === 'around') {
          id = this.maxRange - this.selectedValue - 16;
          if (id < 0) {
            id += this.maxRange
          }
        }

        return `${MEDIA_ROOT}/${this.selectedModel}/${this.selectedOrbit}_${id}_${(+this.shMul).toFixed(2)}.jpg`
      }
    },
    mounted() {
      const scope = this;
    }
  }
</script>

<style lang="scss">
  @import "assets/style.scss";
</style>