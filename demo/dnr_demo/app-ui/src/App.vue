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
                    <div class="gallery-item" v-for="item in examples" v-bind:key="item" @click="selectExample(item)">
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
          <section id="examples" class="examples-section">
            <div id="example-images">
              <div class="image-section with-face-relight-output">
                <div class="image-wrap">
                  <button class="run-functions-button hide">...</button>
                  <img class="target-image" :src="getSelectedModel.active" @click="selectOrbit('none')">

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
                          </button> | 
                          <button class="button" @click="selectOrbit('over')" :class="{selected:selectedOrbit == 'over'}">
                            <span class="sm-text">OVR</span>
                            <span class="m-text">Over</span>
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
  import axios from 'axios';
  const range = {
    'horizontal': {
      minRange: 0,
      maxRange: 70,
      midRange: 34
    },
    'over': {
      minRange: 0,
      maxRange: 34,
      midRange: 16
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
        examples: ['sample_AJ', 'sample_a1', 'sample_aa', 'sample_Faycey', 'sample_kat', 'sample_mal'],
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
            this.selectExample(results.data.filename);
            this.isBusy = false;
          })
          .catch((results) => {
            console.log('FAILURE!!');
            this.isBusy = false;
          });
        }
      }
    },
    computed: {
      getSelectedModel() {
        const defaultURL = `${MEDIA_ROOT}/${this.selectedModel}/ori.jpg`;
        const id = this.selectedOrbit=='over' ? this.maxRange - this.selectedValue : this.selectedValue;
        return {
          active: this.selectedOrbit=='none' ? defaultURL : `${MEDIA_ROOT}/${this.selectedModel}/${this.selectedOrbit}_${id}.jpg`
        }
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