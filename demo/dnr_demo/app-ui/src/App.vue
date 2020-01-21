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
                  <input type="file" class="hidden_input" ref="file" @change="handleFileUpload()" accept="image/png, image/jpeg, image/jpg"/>
                </label>
              </li>
              <li class="nav-list-item divider">-----</li>
              <li class="nav-list-item">
                <div>
                  <h3>Examples</h3>
                  <div class="gallery">
                    <div class="gallery-item" v-for="item in examples" :key="item" @click="selectExample(item)">
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
                  <button class="run-functions-button hide">...</button>

                  <img class="target-image" :src="imgURL" :id="imgIndex" :class="{'hide': imgIndex!=selectedValue}" @click="selectOrbit('none')" v-for="(imgURL, imgIndex) in getSelectedModelRange" :key="imgURL">
                  <div class="image-wrap-cover" :class="{'show': isBusy}">{{isBusy}}</div>
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
                          <input type="range" class="control-slider" 
                            :min="minRange" :max="maxRange" step="1" v-model="selectedValue"
                          >
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
    props: ['id'],
    data() {
      return {
        examples: ['sample_grlee', 'sample_paris','sample_AJ', 'sample_a1', 'sample_aa', 'sample_Faycey', 'sample_kat', 'sample_mal'],
        selectedModel: 'sample_AJ',
        selectedOrbit: defaultOrbit,
        selectedValue: range[defaultOrbit].midRange,
        minRange: range[defaultOrbit].minRange,
        maxRange: range[defaultOrbit].maxRange,
        isMenuShow: false,
        isBusy: false
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
      },
      selectExample(id) {
        this.selectedModel = id
        this.selectOrbit('none')
        this.isMenuShow = false
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
              this.selectExample(results.data.filename);
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
          .catch((results) => {
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
                
        return this.selectedOrbit==='none' ? defaultURL : `${MEDIA_ROOT}/${this.selectedModel}/${this.selectedOrbit}_${id}.jpg`
      }
    },
    computed: { 
      getSelectedModelRange() {        
        const list = new Array(this.maxRange + 1)

        for (let val=this.minRange ; val <= this.maxRange; val++) {
          list[val] = this.generateImageURL(val)
        }

        console.log(list)
        return list
      },
      // getSelectedModel() {
      //   const defaultURL = `${MEDIA_ROOT}/${this.selectedModel}/ori.jpg`;
      //   let id = this.selectedOrbit==='over' ? this.maxRange - this.selectedValue : this.selectedValue;
      //   if (this.selectedOrbit === 'around') {
      //     id = this.maxRange - this.selectedValue - 16;
      //     if (id < 0) {
      //       id += this.maxRange
      //     }
      //   }
                
      //   return {
      //     active: this.selectedOrbit==='none' ? defaultURL : `${MEDIA_ROOT}/${this.selectedModel}/${this.selectedOrbit}_${id}.jpg`
      //   }
      // }
    },
    mounted() {
      const scope = this;
    }
  }
</script>

<style lang="scss">
  @import "assets/style.scss";
</style>